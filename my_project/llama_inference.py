import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import time
import csv
from pathlib import Path
import argparse
import datasets
import traceback
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv
from typing import Optional

try:
    # huggingface-cli login 토큰(로컬 캐시)에서 토큰을 읽기 위함
    from huggingface_hub import get_token as hf_get_token
except Exception:
    hf_get_token = None

from config import Config, ArchType, DataTypeLlama
import llama1
import llama2
import arch_util as archutil
import tmpl_util as tmplutil

class LLAMAWork:
    """
    ArchType, Config, DataType을 파라미터로 받는 최종 워크로드 클래스.
    """
    def __init__(self, arch_type: ArchType, data_type: DataTypeLlama, config: Config, device):
        self.ctx_window_dec = config.ctx_window_dec

        self.arch_type = arch_type
        self.data_type = data_type
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
    
        self._setup_model()

    def _setup_model(self):
        """arch_type과 config 객체를 기반으로 모델을 준비합니다."""
        # -------------------- env setting
        # NOTE: my_project는 worldclass-2025와 폴더 트리가 달라서,
        # "부모 디렉토리에서 .env를 찾는" 방식만으로는 원래 .env를 못 찾을 수 있습니다.
        # - 해결 1) DOTENV_PATH=/path/to/.env 로 명시
        # - 해결 2) huggingface-cli login 토큰 캐시 사용(아래 토큰 로더에서 처리)
        def _resolve_dotenv_path() -> Optional[Path]:
            explicit = os.getenv("DOTENV_PATH")
            if explicit:
                p = Path(explicit).expanduser()
                try:
                    p = p.resolve()
                except Exception:
                    pass
                return p if p.exists() else None

            try:
                here = Path(__file__).resolve()
                for p in [here.parent, *here.parents]:
                    candidate = p / ".env"
                    if candidate.exists():
                        return candidate
            except Exception:
                pass
            return None

        dotenv_path_used = _resolve_dotenv_path()
        if dotenv_path_used is not None:
            load_dotenv(dotenv_path=dotenv_path_used)
            print(f"✅ .env 로드: {dotenv_path_used}")
        else:
            load_dotenv()
            print("⚠️ 경고: .env 파일을 찾지 못했습니다. 필요하면 DOTENV_PATH로 경로를 지정하세요.")

        tokenizer_id = ""
        # -------------------- toeknizer setting
        if self.arch_type == ArchType.LLAMA1 :
            tokenizer_id = "huggyllama/llama-7b"
                
        elif self.arch_type == ArchType.LLAMA2 :
            tokenizer_id = "meta-llama/Llama-2-7b-hf"

        elif self.arch_type == ArchType.LLAMA3 :
            tokenizer_id = "meta-llama/Meta-Llama-3-8B"

        if tokenizer_id:
            try:
                # Hugging Face 토큰 env var는 환경마다 이름이 다를 수 있습니다.
                # - HUGGINGFACE_HUB_TOKEN (공식)
                # - HF_TOKEN (일부 예제/환경)
                # - HUGGING_FACE_HUB_TOKEN (이 프로젝트에서 쓰던 이름)
                hf_token = (
                    os.getenv("HUGGINGFACE_HUB_TOKEN")
                    or os.getenv("HF_TOKEN")
                    or os.getenv("HUGGING_FACE_HUB_TOKEN")
                )
                # huggingface-cli login 이 되어 있으면, 로컬 캐시 토큰을 자동으로 사용 가능
                if not hf_token and hf_get_token is not None:
                    try:
                        hf_token = hf_get_token()
                    except Exception:
                        hf_token = None

                if hf_token:
                    print(f"✅ 토큰이 성공적으로 로드되었습니다: {str(hf_token)[:5]}...")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=hf_token)
                    except TypeError:
                        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_auth_token=hf_token)
                else:
                    print("⚠️ 경고: Hugging Face 토큰이 설정되지 않았습니다. (예: HUGGINGFACE_HUB_TOKEN)")
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

                # Llama 토크나이저는 pad_token이 없는 경우가 많아 eos_token으로 설정
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                print(f"오류: 토크나이저 다운로드에 실패했습니다. - {e}")
                print(traceback.format_exc())
                self.tokenizer = None

        # -------------------- vocab size sanity
        # CUDA device-side assert(embedding index out of range)의 대표 원인:
        # - config.vocab_size < tokenizer가 만들어내는 token id 범위
        if self.tokenizer is not None:
            tok_vocab_size = len(self.tokenizer)
            if self.config.vocab_size != tok_vocab_size:
                print(
                    f"⚠️ 경고: config.vocab_size({self.config.vocab_size}) != tokenizer_vocab_size({tok_vocab_size}). "
                    f"CUDA 인덱스 오류 방지를 위해 vocab_size를 tokenizer에 맞춥니다."
                )
                self.config.vocab_size = tok_vocab_size

        # -------------------- model setting (tokenizer/vocab_size 확정 후)
        if self.arch_type == ArchType.LLAMA1:
            ModelClass = llama1.LLaMA1
        elif self.arch_type in [ArchType.LLAMA2, ArchType.LLAMA3]:
            ModelClass = llama2.LLaMA2
        else:
            raise ValueError(f"지원하지 않는 아키텍처 타입입니다: {self.arch_type.name}")

        try:
            self.model = ModelClass(self.config).to(self.device)
            self.model.eval()
            print("✅ 모델이 성공적으로 GPU에 로드되었습니다.")
        except torch.cuda.OutOfMemoryError:
            print("cuda OOM")
            self.model = None # 진행 불가 상태임을 명시
        except Exception as e:
            print(f"❌ 모델 설정 중 예상치 못한 오류 발생: {e}")
            print(traceback.format_exc())
            self.model = None
        
        # -------------------- mask setting
        # self.max_len = self.config.ctx_window_dec
        # mask = torch.triu(torch.ones(self.max_len, self.max_len, device=self.device), diagonal=1).bool()
        # self.causal_mask = mask

    def _load_data(self, path, name, split):
        """Load our dataset for inference.

        The size of a work dataset must be bigger than batch size for proper
        experiments, and we carefully set that the dataset size of work is
        `brain.work.tmpl.util.INF_DATASET_SIZE`. In this respect, the proper
        data split to use differs across datasets.
        """
        return datasets.load_dataset(path, name, split=f"{split}[:{tmplutil.INF_DATASET_SIZE}]")

    def _prepare_data(self):
        """
        DataType enum에 따라 지정된 데이터셋을 로드하고 모델 입력으로 처리합니다.
        """
        if self.tokenizer is None:
            raise ValueError("오류: 데이터 처리를 위해 tokenizer가 필요합니다.")

        path, config, split = None, None, "test"  # 기본값 설정
    
        def format_prompt(sample):
            """각 샘플을 프롬프트 문자열로 변환하는 함수"""
            # FIXME: do case-analysis properly
            if self.data_type == DataTypeLlama.MMLU:
                choices = "".join([f"{chr(65+i)}. {choice}\n" for i, choice in enumerate(sample['choices'])])
                return f"Question: {sample['question']}\n\nChoices:\n{choices}Answer:"
            elif self.data_type in [DataTypeLlama.TRIVIAQA, DataTypeLlama.NQ]:
                return f"Question: {sample['question']}\nAnswer:"
            elif self.data_type == DataTypeLlama.GSM8K:
                return f"Question: {sample['question']}\nLet's think step by step."
            return ""

        if self.data_type == DataTypeLlama.MMLU:
            path, config, split = "cais/mmlu", "professional_law", "test"
        elif self.data_type == DataTypeLlama.TRIVIAQA:
            path, config, split = "mandarjoshi/trivia_qa", "rc.nocontext", "test"
        elif self.data_type == DataTypeLlama.NQ:
            path, config, split = "google-research-datasets/nq_open", None, "validation"
        elif self.data_type == DataTypeLlama.GSM8K:
            path, config, split = "openai/gsm8k", "main", "test"
        else:
            raise NotImplementedError(f"'{self.data_type.name}' 데이터셋 처리 로직이 구현되지 않았습니다.")

        def tokenize_fn(sample):
            prompt_text = format_prompt(sample)
            toks = self.tokenizer(
                prompt_text,
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=True,
                max_length=self.config.ctx_window_dec,
            )
            return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}

        try:
            raw_dataset = self._load_data(path, name=config, split=split)
            # Avoid CUDA re-init in forked subprocesses: run map sequentially
            processed_inputs = raw_dataset.map(
                tokenize_fn,
                remove_columns=raw_dataset.column_names,
                desc=f"Tokenizing {self.data_type.name}",
                num_proc=None
            )

        except Exception as e:
            print(f"{self.data_type.name} 데이터셋 처리 중 오류 발생: {e}")
            # Fallback: process sequentially with simple loop
            try:
                processed_inputs = []
                for sample in tqdm(raw_dataset, desc=f"Processing {self.data_type.name} (fallback)"):
                    toks = self.tokenizer(
                        format_prompt(sample),
                        add_special_tokens=True,
                        return_attention_mask=True,
                        truncation=True,
                        max_length=self.config.ctx_window_dec,
                    )
                    processed_inputs.append({
                        "input_ids": toks["input_ids"],
                        "attention_mask": toks["attention_mask"],
                    })
            except Exception as e2:
                print(f"{self.data_type.name} 데이터셋 처리 fallback 도 실패: {e2}")
                return None

        print(f"--- 데이터 준비 완료: 총 {len(processed_inputs)}개 샘플 처리 ---")
        class CustomTextDataset(Dataset):
            def __init__(self, tokenized_data):
                self.data = tokenized_data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                item = self.data[idx]
                return torch.tensor(item["input_ids"], dtype=torch.long), torch.tensor(item["attention_mask"], dtype=torch.long)

        
        tensor_dataset = CustomTextDataset(processed_inputs)
        def collate_fn(batch):
            """
            길이가 다른 텐서들을 묶어 패딩을 추가하고, 어텐션 마스크를 생성합니다.
            """
            ids, attn = zip(*batch)
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id).long()
            attention_mask = pad_sequence(attn, batch_first=True, padding_value=0).long()
            return input_ids, attention_mask

        data_loader = DataLoader(
            tensor_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
            shuffle=False,  # 추론 시에는 보통 섞지 않음
            num_workers=0,           # 메모리 복제 방지
            pin_memory=False, 
        )
        return processed_inputs, data_loader

    def run_speed_benchmark(self, prompt_length=100):
        """
        토큰 생성 속도를 측정합니다.
        """
        if prompt_length >= self.config.ctx_window_dec:
            print("오류: 초기 프롬프트 길이가 최대 길이보다 같거나 깁니다.")
            return None

        print(f"--- 최대 {self.config.ctx_window_dec}개 토큰, 초기 입력 {prompt_length}개로 속도 측정 시작 ---")
        self.model.eval() 

        generated_ids = torch.arange(prompt_length, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            start_time = time.time()
            
            for _ in tqdm(range(self.config.ctx_window_dec - prompt_length - 1), desc="토큰 생성 중"):
                seq_len = generated_ids.shape[1]
                causal = torch.triu(torch.ones(seq_len, seq_len, device=self.device), 1).bool()
                mask = causal[:seq_len, :seq_len]
                # GPT 모델은 일반적으로 logits만 반환
                logits = self.model(generated_ids, mask=mask)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

            end_time = time.time()

        total_time = end_time - start_time
        final_shape = generated_ids.shape
        num_generated = final_shape[1] - prompt_length
        tokens_per_sec = num_generated / total_time if total_time > 0 else 0
        
        return {
            "total_time": total_time,
            "tokens_generated": num_generated,
            "tokens_per_sec": tokens_per_sec,
            "final_shape": final_shape
        }
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, max_new_tokens: int = 256, stop_at_eos: bool = True):
        """
        하나의 입력(input_ids)에 대해 추론을 수행하여 새로운 토큰을 생성합니다.

        :param input_ids: 토큰화된 프롬프트 텐서. (예: tensor([[1, 2, 3]]))
        :param max_new_tokens: 생성할 최대 토큰 수
        :param stop_at_eos: EOS 토큰을 만나면 생성을 중단할지 여부
        :return: (생성된 전체 토큰 ID 텐서, 디코딩된 전체 텍스트)
        """
        self.model.eval()
        

        if attention_mask is not None:
            actual_lengths = attention_mask.sum(dim=1)
            trimmed_inputs = []
            for i in range(input_ids.shape[0]):
                trimmed_inputs.append(input_ids[i, -actual_lengths[i]:])
            # pad_sequence로 다시 배치로 만듦
            input_ids = pad_sequence(trimmed_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            # attention_mask도 모두 1로 (패딩 없음)
            attention_mask = torch.ones_like(input_ids)

        generated_ids = input_ids.to(self.device)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                generated_ids = archutil.crop_data_to_ctx_window(generated_ids, self.ctx_window_dec)

                batch_size, seq_len = generated_ids.shape

                # mask = self.causal_mask[:seq_len, :seq_len]
                mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), 1).bool()
                mask = mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
                logits = self.model(generated_ids, mask=mask)
                
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                # eos 토큰 생성 시 중지 코드 삭제
                # eos_id = self.tokenizer.eos_token_id
                # nt = next_token
                # if not isinstance(nt, torch.Tensor):
                #     nt = torch.tensor(nt, device=generated_ids.device)
                # nt_flat = nt.flatten()
                # result = (nt_flat == eos_id)
                # if isinstance(result, torch.Tensor):
                #     if result.numel() == 1:
                #         if result.item():
                #             break
                #     else:
                #         if result.all():
                #             break
                # else:
                #     if result:
                #         break

        # decoded_text = self.tokenizer.decode(generated_ids[0].tolist())
        # decoded_texts = [self.tokenizer.decode(ids.tolist()) for ids in generated_ids]
        return generated_ids#, decoded_texts
        

    def run_dataset_inference(self, data_loader, output_csv_path, max_new_tokens_per_sample=50):
        """
        준비된 전체 데이터셋에 대해 추론을 수행하고, 성능을 측정합니다.

        :param prepared_data: _prepare_data에서 반환된 토큰화된 프롬프트 텐서 리스트
        :param max_new_tokens_per_sample: 각 샘플마다 생성할 최대 토큰 수
        """
        total_samples = len(data_loader.dataset)
        total_tokens_generated = 0
        processed_samples = 0

        results_list = []
        csv_file = None
        csv_writer = None
        if output_csv_path:
            try:
                # 파일을 쓰기 모드로 열고, csv writer 준비
                csv_file = open(output_csv_path, 'w', newline='', encoding='utf-8')
                csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL, escapechar='\\')
                # CSV 헤더 작성
                csv_writer.writerow(['input_prompt', 'generated_output'])
                print(f"\n--- 추론 결과를 '{output_csv_path}' 파일에 저장합니다. ---")
            except IOError as e:
                print(f"⚠️ 경고: CSV 파일을 열 수 없습니다. - {e}")
                output_csv_path = None # 저장 기능 비활성화

        
        # 전체 추론 시간을 측정하기 위한 타이머
        # overall_start_time = time.time()
        use_cuda_timing = (isinstance(self.device, str) and self.device.startswith("cuda") and torch.cuda.is_available())
        gpu_time_total = 0.0

        for input_ids_batch, attention_mask_batch in tqdm(data_loader, desc="Dataset Batch Inference"):
            try :
                # Prevent CUDA device-side assert (embedding index out of range)
                max_id = int(input_ids_batch.max().item()) if hasattr(input_ids_batch, "max") else None
                if max_id is not None and max_id >= int(self.config.vocab_size):
                    raise ValueError(
                        f"input_ids에 vocab 범위를 벗어난 토큰이 있습니다. "
                        f"max_token_id={max_id}, config.vocab_size={self.config.vocab_size}. "
                        f"(보통 tokenizer vocab과 model vocab_size 불일치)"
                    )

                if use_cuda_timing:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                generated_ids = self.generate(
                    input_ids_batch, 
                    attention_mask=attention_mask_batch,  # 모델이 지원하면
                    max_new_tokens=max_new_tokens_per_sample
                )
                if use_cuda_timing:
                    end_event.record(); torch.cuda.synchronize()
                    gpu_time_total += start_event.elapsed_time(end_event) / 1000.0

                decoded_texts = [self.tokenizer.decode(ids.cpu().tolist()) for ids in generated_ids]
                prompt_lengths = attention_mask_batch.sum(dim=1)
                prompt_texts = [
                    self.tokenizer.decode(ids[:plen].tolist(), skip_special_tokens=True)
                    for ids, plen in zip(input_ids_batch, prompt_lengths)
                ]
                for prompt, full_output in zip(prompt_texts, decoded_texts):
                    if csv_writer:
                        csv_writer.writerow([prompt, full_output])

            except torch.cuda.OutOfMemoryError:
                print("cuda OOM")
                torch.cuda.empty_cache()
                if csv_file:
                    csv_file.close()
                return None
            except Exception as e:
                print(f"❌ 추론 실행 중 예상치 못한 오류 발생: {e}")
                return None

        if csv_file:
            csv_file.close()

        total_tokens_generated_val = total_tokens_generated.item() if isinstance(total_tokens_generated, torch.Tensor) else total_tokens_generated
    
        tokens_per_sec = total_tokens_generated_val / gpu_time_total if gpu_time_total > 0 else 0
        avg_time_per_sample = gpu_time_total / processed_samples if processed_samples > 0 else 0

        results = {
            "dataset_name": self.data_type.name,
            "processed_samples": processed_samples,
            "total_samples": total_samples,
            "total_tokens_generated": total_tokens_generated,
            "total_inference_time_sec": round(gpu_time_total, 2),
            "avg_time_per_sample_sec": round(avg_time_per_sample, 3),
            "tokens_per_sec": round(tokens_per_sec, 2)
        }
        
        print("--- 측정 완료 ---")
        return results

def run(archtype, datatype, config, cuda_idx, seed):
    tmplutil.set_seed(seed)

    # work
    work_runner = LLAMAWork(arch_type=archtype, data_type=datatype, config=config, device=f"cuda:{cuda_idx}")
    if work_runner.model is None:
        print("모델 로딩에 실패하여 프로그램을 종료합니다.")
        return

    # data
    prepared_data, data_loader = work_runner._prepare_data()
    if data_loader is None:
        print("데이터 준비에 실패하여 프로그램을 종료합니다.")
        return

    # inference
    results = work_runner.run_dataset_inference(data_loader,output_csv_path='./res_llama_inference.csv')
    if results is None:
        print("\n추론 과정에서 오류가 발생하여 벤치마크를 중단했습니다.")
    else:
        print("\n[최종 측정 결과]:", results)

    # Memory management: make sure to delete references to the objects
    del work_runner, prepared_data, data_loader, results

if __name__ == "__main__":
    """
    # == llama-3
    datatype = all_members_list[0]
    archtype = ArchType.LLAMA3
    config = Config(
        vocab_size=128256,
        batch_size=8,
        ctx_window_enc=8192,
        ctx_window_dec=8192,
        d_emb=4096,
        d_q=128,
        d_k=128,
        d_v=128,
        d_ff=14336,
        n_heads_enc=0,
        n_heads_dec_sa=32,
        n_heads_dec_ca=0,
        n_layers_enc=0,
        n_layers_dec=32,
        dropout_rate_enc=0.1,
        dropout_rate_dec=0.1
    )
    run(archtype, datatype, config, cuda_idx=0, seed=2025)

    # == llama-2
    archtype = ArchType.LLAMA2
    datatype = DataTypeLlama.MMLU
    config = Config(
        vocab_size=32000,
        batch_size=16,
        ctx_window_enc=4096,
        ctx_window_dec=4096,
        d_emb=4096,
        d_q=128,
        d_k=128,
        d_v=128,
        d_ff=11008,
        n_heads_enc=0,
        n_heads_dec_sa=32,
        n_heads_dec_ca=0,
        n_layers_enc=0,
        n_layers_dec=32,
        dropout_rate_enc=0.1,
        dropout_rate_dec=0.1
    )
    """

    # == llama-3
    # archtype = ArchType.LLAMA3
    # datatype = DataTypeLlama.MMLU
    # config = Config(
    #     vocab_size=128256,
    #     batch_size=8,
    #     ctx_window_enc=8192,
    #     ctx_window_dec=8192,
    #     d_emb=4096,
    #     d_q=128,
    #     d_k=128,
    #     d_v=128,
    #     d_ff=14336,
    #     n_heads_enc=0,
    #     n_heads_dec_sa=32,
    #     n_heads_dec_ca=0,
    #     n_layers_enc=0,
    #     n_layers_dec=32,
    #     dropout_rate_enc=0.1,
    #     dropout_rate_dec=0.1
    # )

    # run(archtype, datatype, config, cuda_idx=0, seed=2025)

    all_members_list = list(DataTypeLlama)
    # for member in all_members_list:
    archtype = ArchType.LLAMA1
    datatype = DataTypeLlama.MMLU
    config = Config(
        vocab_size=32000,
        batch_size=12,
        ctx_window_enc=256,
        ctx_window_dec=256,
        d_emb=4096,
        d_q=128,
        d_k=128,
        d_v=128,
        d_ff=110,
        n_heads_enc=0,
        n_heads_dec_sa=8,
        n_heads_dec_ca=0,
        n_layers_enc=0,
        n_layers_dec=8,
        dropout_rate_enc=0.1,
        dropout_rate_dec=0.1
    )
    run(archtype, datatype, config, cuda_idx=2, seed=2025)

    

