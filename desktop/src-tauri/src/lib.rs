// SAFIR desktop shell. The Rust side is intentionally thin in this step: it
// hosts the Nuxt webview and enables the dialog plugin (native file picker for
// selecting a local test video). All analysis logic stays in the FastAPI
// backend — the desktop app only talks to it over HTTP/SSE.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("error while running SAFIR desktop application");
}
