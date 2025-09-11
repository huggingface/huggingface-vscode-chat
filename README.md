## 🤗 Hugging Face Provider for GitHub Copilot Chat

![Demo](assets/demo.gif)

Bring thousands of open‑source models to GitHub Copilot Chat with a first‑class provider powered by [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) and built on the [VS Code Language Model Chat Provider API](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider).

### ✨ Why use the Hugging Face provider in GitHub Copilot Chat
- 4k+ open‑source LLMs with tool calling capabilities.
- Single API to thousands of open‑source LLMs via providers like Groq, Cerebras, Together AI, SambaNova, and more.
- Built for high availability and low latency through world‑class providers.
- No extra markup on provider rates.

---

## ⚡ Quick start
1. Open VS Code's chat interface.
2. Click the model picker and click "Manage Models...".
3. Select "Hugging Face" provider.
4. Provide your Hugging Face Token, you can get one in your [settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained).
5. Select the models you want to add to the model picker.

💡 The free tier gives you monthly inference credits to start building and experimenting. Upgrade to [Hugging Face PRO](https://huggingface.co/pro) for even more flexibility, $2 in monthly credits plus pay‑as‑you‑go access to all providers!


---

## 🛠️ Development
```bash
git clone https://github.com/huggingface/huggingface-vscode-chat
cd huggingface-vscode-chat
npm install
npm run compile
```
Press F5 to launch an Extension Development Host.

Common scripts:
- Build: `npm run compile`
- Watch: `npm run watch`
- Lint: `npm run lint`
- Format: `npm run format`

---

## 📚 Learn more
- Inference Providers documentation: https://huggingface.co/docs/inference-providers/index
- VS Code Chat Provider API: https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider

---

## Support & License
- Open issues: https://github.com/huggingface/huggingface-vscode-chat/issues
- License: MIT License Copyright (c) 2025 Hugging Face
