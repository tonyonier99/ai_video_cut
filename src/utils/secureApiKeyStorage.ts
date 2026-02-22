const API_KEY_STORAGE_KEY = 'antigravity_api_key';

function isElectronEnvironment(): boolean {
    return typeof window !== 'undefined'
        && window.electronAPI !== undefined
        && window.electronAPI.isElectron === true;
}

export async function loadApiKey(): Promise<string> {
    if (isElectronEnvironment()) {
        const result = await window.electronAPI!.secureStorage.load(API_KEY_STORAGE_KEY);
        if (result.success) {
            return result.value || '';
        }
        // Encryption unavailable — return empty rather than falling back to plaintext
        return '';
    }
    // Browser-only fallback (non-Electron)
    return sessionStorage.getItem(API_KEY_STORAGE_KEY) || '';
}

export async function saveApiKey(value: string): Promise<void> {
    if (isElectronEnvironment()) {
        const result = await window.electronAPI!.secureStorage.save(API_KEY_STORAGE_KEY, value);
        if (!result.success) {
            throw new Error(`Secure storage failed: ${result.error}`);
        }
        return;
    }
    // Browser-only fallback (non-Electron)
    if (value) {
        sessionStorage.setItem(API_KEY_STORAGE_KEY, value);
    } else {
        sessionStorage.removeItem(API_KEY_STORAGE_KEY);
    }
}

export async function deleteApiKey(): Promise<void> {
    if (isElectronEnvironment()) {
        try {
            await window.electronAPI!.secureStorage.delete(API_KEY_STORAGE_KEY);
        } catch {
            // Ignore errors
        }
    }
    sessionStorage.removeItem(API_KEY_STORAGE_KEY);
    localStorage.removeItem(API_KEY_STORAGE_KEY);
}
