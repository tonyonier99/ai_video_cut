const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    secureStorage: {
        save: (key, value) => ipcRenderer.invoke('secure-storage:save', key, value),
        load: (key) => ipcRenderer.invoke('secure-storage:load', key),
        delete: (key) => ipcRenderer.invoke('secure-storage:delete', key),
    },
    isElectron: true,
});
