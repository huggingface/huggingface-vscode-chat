# Hugging Face Inference Providers Chat Extension for VS Code

This extension contributes a Chat Model Provider powered by [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) for VS Code Chat.

This is based on the VS Code Language Model Chat Provider API. Learn more 👉 [here](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider)

## Getting Started

### Prerequisites
- VS Code 1.104 or higher. For dev, it's recommended to install [VS Code Insiders](https://code.visualstudio.com/insiders/).
- Node.js and `npm` (or `pnpm`) installed

### Installation and Development
1. Clone this repository
2. Navigate to the extension directory:
   ```bash
   cd huggingface-vscode-chat
   ```
3. Install dependencies:
   ```bash
   npm install
   ```
4. Compile the extension:
   ```bash
   npm run compile
   ```
5. Press F5 to launch a new Extension Development Host window
6. The extension will be active and ready to provide chat models

### Building and Watching
- Build once: `npm run compile`
- Watch mode: `npm run watch` (automatically recompiles on file changes)
- Lint code: `npm run lint`


## Usage

Once the extension is active:

- Open VS Code's chat interface.
- Click the model picker and click "Manage Models..."
- Select the Hugging Face provider.
- Provide your Hugging Face token.
- Select the models you want to add to the model picker.
- You're all set!