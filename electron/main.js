import { app, BrowserWindow, Menu, ipcMain, safeStorage } from 'electron';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
let pythonProcess = null;

function startPythonBackend() {
    const isDev = !app.isPackaged;
    let command;
    let args;

    if (isDev) {
        command = 'python';
        // Run server.py in the root directory
        const scriptPath = path.join(__dirname, '../server.py');
        args = [scriptPath];
        console.log('Spawning dev backend:', command, args);
    } else {
        // In production, run the bundled executable
        const resourcesPath = process.resourcesPath;
        const executableName = process.platform === 'win32' ? 'antigravity-engine.exe' : 'antigravity-engine';
        const backendPath = path.join(resourcesPath, 'dist/antigravity-engine', executableName);
        console.log('Spawning prod backend from:', backendPath);

        command = backendPath;
        args = [];
    }

    try {
        pythonProcess = spawn(command, args, {
            stdio: 'inherit',
            shell: false
        });

        pythonProcess.on('error', (err) => {
            console.error('Failed to start python backend:', err);
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python backend exited with code ${code}`);
        });
    } catch (e) {
        console.error("Backend spawn error:", e);
    }
}

function createWindow() {
    const win = new BrowserWindow({
        width: 1600,
        height: 900,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            webSecurity: true,
            preload: path.join(__dirname, 'preload.cjs')
        },
        backgroundColor: '#09090b',
        titleBarStyle: 'hiddenInset'
    });

    const isDev = !app.isPackaged;

    if (isDev) {
        win.loadURL('http://localhost:5173');
        // win.webContents.openDevTools();
    } else {
        win.loadFile(path.join(__dirname, '../dist/index.html'));
    }
}

// --- Secure Storage Helpers ---
const ALLOWED_STORAGE_KEYS = new Set(['antigravity_api_key']);
const MAX_VALUE_LENGTH = 1024;

function validateStorageKey(key) {
    if (typeof key !== 'string' || !ALLOWED_STORAGE_KEYS.has(key)) {
        throw new Error(`Invalid storage key: ${key}`);
    }
}

function getStorePath() {
    return path.join(app.getPath('userData'), 'secure-store.json');
}

function readStore() {
    try {
        const raw = fs.readFileSync(getStorePath(), 'utf-8');
        return JSON.parse(raw);
    } catch (err) {
        if (err.code !== 'ENOENT') {
            console.error('Failed to read secure store:', err.message);
        }
        return {};
    }
}

function writeStore(store) {
    fs.writeFileSync(getStorePath(), JSON.stringify(store, null, 2), {
        encoding: 'utf-8',
        mode: 0o600
    });
}

app.whenReady().then(() => {
    // --- Register Secure Storage IPC Handlers ---
    ipcMain.handle('secure-storage:save', (_event, key, value) => {
        try {
            validateStorageKey(key);
            if (typeof value !== 'string' || value.length > MAX_VALUE_LENGTH) {
                return { success: false, error: 'Invalid value' };
            }
            if (!safeStorage.isEncryptionAvailable()) {
                return { success: false, error: 'Encryption not available on this system' };
            }
            const encrypted = safeStorage.encryptString(value);
            const store = { ...readStore(), [key]: encrypted.toString('base64') };
            writeStore(store);
            return { success: true };
        } catch (err) {
            return { success: false, error: err.message };
        }
    });

    ipcMain.handle('secure-storage:load', (_event, key) => {
        try {
            validateStorageKey(key);
            if (!safeStorage.isEncryptionAvailable()) {
                return { success: false, error: 'Encryption not available on this system' };
            }
            const store = readStore();
            const encoded = store[key];
            if (!encoded) {
                return { success: true, value: '' };
            }
            const decrypted = safeStorage.decryptString(Buffer.from(encoded, 'base64'));
            return { success: true, value: decrypted };
        } catch (err) {
            return { success: false, error: err.message };
        }
    });

    ipcMain.handle('secure-storage:delete', (_event, key) => {
        try {
            validateStorageKey(key);
            const store = readStore();
            const { [key]: _removed, ...rest } = store;
            writeStore(rest);
            return { success: true };
        } catch (err) {
            return { success: false, error: err.message };
        }
    });

    // Set custom menu to disable default zoom shortcuts
    const template = [
        ...(process.platform === 'darwin' ? [{
            label: app.name,
            submenu: [
                { role: 'about' },
                { type: 'separator' },
                { role: 'services' },
                { type: 'separator' },
                { role: 'hide' },
                { role: 'hideOthers' },
                { role: 'unhide' },
                { type: 'separator' },
                { role: 'quit' }
            ]
        }] : []),
        {
            label: 'File',
            submenu: [
                { role: 'close' }
            ]
        },
        {
            label: 'Edit',
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'pasteAndMatchStyle' },
                { role: 'delete' },
                { role: 'selectAll' }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: 'Window',
            submenu: [
                { role: 'minimize' },
                { role: 'zoom' },
                ...(process.platform === 'darwin' ? [
                    { type: 'separator' },
                    { role: 'front' },
                    { type: 'separator' },
                    { role: 'window' }
                ] : [
                    { role: 'close' }
                ])
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);

    startPythonBackend();
    setTimeout(createWindow, 1000);

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('will-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
