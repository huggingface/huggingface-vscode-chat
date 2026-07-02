import * as vscode from "vscode";
import { HuggingFaceChatModelProvider } from "./provider";

export function activate(context: vscode.ExtensionContext) {
	// Build a descriptive User-Agent to help quantify API usage
	const ext = vscode.extensions.getExtension("HuggingFace.huggingface-vscode-chat");
	const extVersion = ext?.packageJSON?.version ?? "unknown";
	const vscodeVersion = vscode.version;
	// Keep UA minimal: only extension version and VS Code version
	const ua = `huggingface-vscode-chat/${extVersion} VSCode/${vscodeVersion}`;

	const provider = new HuggingFaceChatModelProvider(context.secrets, ua);
	// Register the Hugging Face provider under the vendor id used in package.json
	vscode.lm.registerLanguageModelChatProvider("huggingface", provider);

	// Management command: configure the API key and optional organization billing.
	context.subscriptions.push(
		vscode.commands.registerCommand("huggingface.manage", async () => {
			await manageProvider(context);
		})
	);
}

/** Present a small menu to manage the token and organization billing target. */
async function manageProvider(context: vscode.ExtensionContext): Promise<void> {
	const config = vscode.workspace.getConfiguration("huggingface");
	const hasKey = !!(await context.secrets.get("huggingface.apiKey"));
	const billTo = (config.get<string>("billTo") ?? "").trim();

	type ManageAction = "setToken" | "setBillTo" | "clearBillTo";
	const items: (vscode.QuickPickItem & { action: ManageAction })[] = [
		{
			label: "$(key) Set Hugging Face Token",
			description: hasKey ? "Update or clear your saved token" : "Not set",
			action: "setToken",
		},
		{
			label: "$(organization) Bill to an Organization",
			description: billTo ? `Currently billing to "${billTo}"` : "Off — usage billed to your account",
			detail: "Charge inference usage to a Team/Enterprise organization or resource group (X-HF-Bill-To)",
			action: "setBillTo",
		},
	];
	if (billTo) {
		items.push({
			label: "$(clear-all) Stop Billing to Organization",
			description: "Bill usage to your personal account",
			action: "clearBillTo",
		});
	}

	const pick = await vscode.window.showQuickPick(items, {
		title: "Manage Hugging Face Provider",
		placeHolder: "Configure your Hugging Face token and billing",
	});
	if (!pick) {
		return; // user dismissed the menu
	}

	switch (pick.action) {
		case "setToken":
			await setToken(context);
			break;
		case "setBillTo":
			await setBillTo(config, billTo);
			break;
		case "clearBillTo":
			await config.update("billTo", "", vscode.ConfigurationTarget.Global);
			vscode.window.showInformationMessage("Hugging Face inference is now billed to your personal account.");
			break;
	}
}

async function setToken(context: vscode.ExtensionContext): Promise<void> {
	const existing = await context.secrets.get("huggingface.apiKey");
	const apiKey = await vscode.window.showInputBox({
		title: "Hugging Face API Key",
		prompt: existing ? "Update your Hugging Face API key (leave empty to clear)" : "Enter your Hugging Face API key",
		ignoreFocusOut: true,
		password: true,
		value: existing ?? "",
	});
	if (apiKey === undefined) {
		return; // user canceled
	}
	if (!apiKey.trim()) {
		await context.secrets.delete("huggingface.apiKey");
		vscode.window.showInformationMessage("Hugging Face API key cleared.");
		return;
	}
	await context.secrets.store("huggingface.apiKey", apiKey.trim());
	vscode.window.showInformationMessage("Hugging Face API key saved.");
}

async function setBillTo(config: vscode.WorkspaceConfiguration, existing: string): Promise<void> {
	const billTo = await vscode.window.showInputBox({
		title: "Bill to a Hugging Face Organization",
		prompt: "Organization name or Enterprise resource group ID to bill inference usage to. Leave empty to bill your personal account.",
		placeHolder: "my-org  •  or a resource group ID",
		ignoreFocusOut: true,
		value: existing,
	});
	if (billTo === undefined) {
		return; // user canceled
	}
	const trimmed = billTo.trim();
	await config.update("billTo", trimmed, vscode.ConfigurationTarget.Global);
	vscode.window.showInformationMessage(
		trimmed
			? `Hugging Face inference will be billed to "${trimmed}".`
			: "Hugging Face inference is now billed to your personal account."
	);
}

export function deactivate() {}
