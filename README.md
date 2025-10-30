# 🤗 Hugging Face Provider for GitHub Copilot Chat

![Demo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/inference-providers-guides/demo_vscode.gif)

Use frontier open LLMs like Kimi K2, DeepSeek V3.1, GLM 4.5 and more in VS Code with GitHub Copilot Chat powered by [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) 🔥

---

## ⚡ Quick Start
1. Install the HF Copilot Chat extension [here](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode-chat).
2. Open VS Code's chat interface.
3. Click the model picker and click "Manage Models...".
4. Select "Hugging Face" provider.
5. Provide your Hugging Face Token, you can get one in your [settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). You only need to give it the inference.serverless permissions.
6. Choose the models you want to add to the model picker. 🥳

Each model entry also offers `cheapest` and `fastest` mode for each model. `fastest` selects the provider with highest throughput and `cheapest` selects the provider with lowest price per output token.

## ✨ Why use the Hugging Face provider in Copilot
* Access [SoTA open-source LLMs](https://huggingface.co/models?pipeline_tag=text-generation&inference_provider=cerebras,together,fireworks-ai,nebius,novita,sambanova,groq,hyperbolic,nscale,fal-ai,cohere,replicate,scaleway,black-forest-labs,ovhcloud&sort=trending) with tool calling capabilities.
* Single API to switch between multiple providers: Cerebras, Cohere, Fireworks AI, Groq, HF Inference, Hyperbolic, Nebius, Novita, Nscale, SambaNova, Together AI, and more. See the full list of partners in the [Inference Providers docs](https://huggingface.co/docs/inference-providers/index#partners).
* Built for high availability (across providers) and low latency.
* Transparent pricing: what the provider charges is what you pay.

💡 The free Hugging Face user tier gives you a small amount of monthly inference credits to experiment. Upgrade to [Hugging Face PRO](https://huggingface.co/pro) or [Enterprise](https://huggingface.co/enterprise) for $2 in monthly credits plus pay-as-you-go access across all providers!

---

## Requirements
* VS Code 1.104.0 or higher.
* Hugging Face access token with `inference.serverless` permissions.

## 🛠️ Development
```bash
git clone https://github.com/huggingface/huggingface-vscode-chat
cd huggingface-vscode-chat
npm install
npm run compile
```
Press F5 to launch an Extension Development Host.

Common scripts:
* Build: `npm run compile`
* Watch: `npm run watch`
* Lint: `npm run lint`
* Format: `npm run format`

---

## 📚 Learn more
* Inference Providers documentation: https://huggingface.co/docs/inference-providers/index
* VS Code Chat Provider API: https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider

---

## Support & License
* Open issues: https://github.com/huggingface/huggingface-vscode-chat/issues
* License: MIT License Copyright (c) 2025 Hugging Face
