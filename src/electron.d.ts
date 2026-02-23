interface SecureStorageResult {
    success: boolean;
    value?: string;
    error?: string;
}

interface ElectronAPI {
    secureStorage: {
        save: (key: string, value: string) => Promise<SecureStorageResult>;
        load: (key: string) => Promise<SecureStorageResult>;
        delete: (key: string) => Promise<SecureStorageResult>;
    };
    isElectron: true;
}

interface Window {
    electronAPI?: ElectronAPI;
}
