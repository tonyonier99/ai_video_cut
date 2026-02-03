# Feature Implementation Plan

## Phase 1: Core Architecture & Asset Management (Completed)
- [x] **Project Structure Refactor**
    - [x] Create distinct "Project" vs "Timeline" concepts in state.
    - [x] Implement left-panel tabs (Project / Controls / Effects).
- [x] **Asset Management**
    - [x] Create `Asset` interface.
    - [x] Build "Project Bin" UI.
    - [x] Implement "Import Media" (Video/Audio/Image).

## Phase 2: Advanced Track Engine (In Progress)
- [x] **Data Model Update**
    - [x] Update `Cut` interface (`trackId`, `assetId`).
    - [x] Add "Text" track support in state (`videoTracks`).
- [x] **Text/Title Engine** (Basic)
    - [x] Implement `TextOverlayRenderer` in Player.
    - [x] Add "Back Text" tool to toolbar.
    - [x] Basic text rendering on screen.
    - [ ] **Rich Text Editing**: Font, Size, Color in Inspector. (Next Step)

## Phase 3: Effects & Polish (Planned)
- [ ] **Transition System**: Cross Dissolve implementation.
- [ ] **Animations**: Keyframe support.
- [ ] **Undo/Redo**: Implement History stack.
- [ ] **Export**: Update Export logic to handle multi-track.
