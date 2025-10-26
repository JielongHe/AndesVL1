---
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: transformers
---

<div align="center">
  <h1>AndesVL-1B-Instruct</h1>
<a href='https://arxiv.org/abs/2510.11496'><img src='https://img.shields.io/badge/arXiv-2510.11496-b31b1b.svg'></a> &nbsp;
<a href='https://huggingface.co/OPPOer'><img src='https://img.shields.io/badge/🤗%20HuggingFace-AndesVL-ffd21f.svg'></a> &nbsp;
<a href='https://github.com/OPPO-Mente-Lab/AndesVL_Evaluation'><img src="https://img.shields.io/badge/GitHub-OPPOer-blue.svg?logo=github" alt="GitHub"></a>
</div>

AndesVL is a suite of mobile-optimized Multimodal Large Language Models (MLLMs) with **0.6B to 4B parameters**, built upon Qwen3's LLM and various visual encoders. Designed for efficient edge deployment, it achieves first-tier performance on diverse benchmarks, including those for text-rich tasks, reasoning tasks, Visual Question Answering (VQA), multi-image tasks, multilingual tasks, and GUI tasks. Its "1+N" LoRA architecture and QALFT framework facilitate efficient task adaptation and model compression, enabling a 6.7x peak decoding speedup and a 1.8 bits-per-weight compression ratio on mobile chips.

Detailed model sizes and components are provided below:

| Model | Total Parameters (B) | Visual Encoder | LLM |
|---|---|---|---|
| AndesVL-0.6B | 0.695 | SigLIP2-Base | Qwen3-0.6B |
| **AndesVL-1B** | 0.927 | AIMv2-Large | Qwen3-0.6B |
| AndesVL-2B | 2.055 | AIMv2-Large | Qwen3-1.7B|
| AndesVL-4B | 4.360 | AIMv2-Large | Qwen3-4B |


# Quick Start
```commandline
# require transformers>=4.52.4

import torch
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

model_dir = "OPPOer/AndesVL-1B-Instruct"

model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
image_processor = CLIPImageProcessor.from_pretrained(model_dir, trust_remote_code=True)

messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "描述这张图片。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://i-blog.csdnimg.cn/blog_migrate/2f4c88e71f7eabe46d062d2f1ec77d10.jpeg" # image/to/path
                            },
                        }
                    ],
                },
        ]
res = model.chat(messages, tokenizer, image_processor, max_new_tokens=1024, do_sample=True, temperature=0.6)
print(res)
```

# Citation
If you find our work helpful, feel free to give us a cite.

```
@misc{jin2025andesvltechnicalreportefficient,
      title={AndesVL Technical Report: An Efficient Mobile-side Multimodal Large Language Model}, 
      author={AndesVL Team, OPPO AI Center},
      year={2025},
      eprint={2510.11496},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.11496}, 
}
```

# Acknowledge
We are very grateful for the efforts of the [Qwen](https://huggingface.co/Qwen), [AimV2](https://huggingface.co/apple/aimv2-large-patch14-224) and [Siglip 2](https://arxiv.org/abs/2502.14786) projects.
