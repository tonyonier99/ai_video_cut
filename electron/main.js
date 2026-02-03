import { app, BrowserWindow } from 'electron';
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
            nodeIntegration: true,
            contextIsolation: false,
            webSecurity: false
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

app.whenReady().then(() => {
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
