# Icons

Replace these with real assets before shipping. Required files:

- `32x32.png`           — small tray / window icon
- `128x128.png`         — Linux desktop / main window
- `128x128@2x.png`      — same at 2× DPI
- `icon.icns`           — macOS bundle icon
- `icon.ico`            — Windows installer icon

Generate from a single SVG with:

```bash
pnpm tauri icon path/to/source.png
```

This populates everything Tauri's bundler needs.
