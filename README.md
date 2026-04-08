# IG Collection to Obsidian

Small Python CLI for turning an Instaloader `:saved` archive into Obsidian notes with linked media, saved-collection tags, optional Faster-Whisper transcripts, and optional AI categorization.

## What It Does

- Runs `instaloader` against your saved Instagram posts if you want the project to own the download step.
- Scans an existing Instaloader archive and groups files into posts by shortcode.
- Reads adjacent caption text, transcript files, and JSON metadata when they exist.
- Applies collection labels from a local `collections.json`.
- Extracts locations from Instagram metadata or a manual `locations.json`.
- Classifies posts into your own taxonomy using caption/transcript text when AI categorization is enabled.
- Symlinks or copies media into an Obsidian-friendly folder.
- Writes one markdown note per post/reel with YAML frontmatter and embedded media links.
- Exports a Google My Maps compatible CSV with names, descriptions, and coordinates/addresses.

## Project Layout

```text
archive/
  saved/
    someuser/
      2026-04-08_12-01-02_ABC123.mp4
      2026-04-08_12-01-02_ABC123.txt
      2026-04-08_12-01-02_ABC123.json.xz
      2026-04-08_12-01-02_ABC123.transcript.txt

vault/
  instagram/
    notes/
      2026-04-08_ABC123.md
    media/
      someuser/
        2026-04-08_12-01-02_ABC123.mp4
    exports/
      google-my-maps.csv
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[transcribe,categorize]'
cp config.example.json config.json
cp collections.example.json collections.json
cp locations.example.json locations.json
cp taxonomy.example.json taxonomy.json
```

If you do not need editable install mode, this also works:

```bash
pip install -r requirements.txt
```

Edit `config.json`:

- `paths.archive_dir`: your Instaloader output folder, usually `~/archive/instagram/saved`
- `paths.vault_dir`: the Obsidian folder you want this tool to manage
- `download.instagram_username`: your Instagram username if you want `ig-obsidian download` / `sync --download` to run Instaloader for you
- `paths.locations_file`: optional per-shortcode location overrides and descriptions for Google Maps export
- `categorization.taxonomy_file`: your category taxonomy for AI classification

## Collection Mapping

`collections.json` supports either shape:

```json
{
  "ABC123": "travels",
  "XYZ789": "food"
}
```

or:

```json
{
  "travels": ["ABC123", "https://www.instagram.com/reel/XYZ789/"],
  "food": ["LMN456"]
}
```

If you do not have an automated way to export Instagram collections yet, keep this file manual. That was the weak point in the original plan too.

## Location Mapping

`locations.json` is optional, but it is the cleanest way to add descriptions and fix missing coordinates. It is keyed by shortcode and can contain one location object or a list:

```json
{
  "ABC123": {
    "name": "Cafe Example",
    "description": "Good brunch spot from a saved reel.",
    "address": "123 Example Street, Lisbon, Portugal",
    "latitude": 38.7223,
    "longitude": -9.1393
  }
}
```

If Instagram metadata already contains a location, the tool will pick it up automatically. Manual overrides win when both exist.

## AI Taxonomy

`taxonomy.json` defines the categories the model is allowed to use. It can be a list of names or an object of `name -> description`. Descriptions produce better results.

```json
{
  "travel": "Destinations, itineraries, landmarks, hotels, flights, or general trip inspiration.",
  "food": "Restaurants, cafes, recipes, dishes, bakeries, bars, or local food recommendations.",
  "shopping": "Markets, stores, boutiques, malls, and products worth buying."
}
```

When categorization is enabled, the tool:

- reads caption/transcript/location context
- classifies into your taxonomy with OpenAI
- writes a `.ai_categories.json` sidecar next to the source files
- skips already-classified posts on future reruns unless `overwrite` is true
- writes the chosen categories into note frontmatter and an `## AI Categories` section

## Usage

Run the full pipeline:

```bash
ig-obsidian sync --config config.json
```

Force a fresh download first:

```bash
ig-obsidian sync --config config.json --download
```

Only transcribe videos:

```bash
ig-obsidian transcribe --config config.json
```

Only rebuild notes and media links:

```bash
ig-obsidian build --config config.json
```

Only run AI categorization:

```bash
ig-obsidian categorize --config config.json
```

Export locations for Google My Maps:

```bash
ig-obsidian export-maps --config config.json
```

Only run Instaloader:

```bash
ig-obsidian download --config config.json
```

## Notes

- `download` shells out to the `instaloader` CLI. The package is installed as a dependency, but the login flow is still interactive.
- Transcription only supports `faster-whisper` right now. That keeps the implementation simpler and avoids the extra `ffmpeg` requirement from `openai-whisper`.
- The first transcription run may pause on model initialization while `faster-whisper` downloads model files from Hugging Face. The CLI now prints that step explicitly.
- AI categorization currently supports OpenAI via the Responses API and needs `OPENAI_API_KEY` set in your environment.
- Media is mirrored into the Obsidian folder using symlinks by default. Switch `obsidian.use_symlinks` to `false` if you want copies instead.
- Notes are regenerated deterministically. Re-running `build` or `sync` updates the markdown in place.
- Posts are deduped by Instagram shortcode, and per-post media variants are deduped by their sidecar slot so reruns do not create extra notes for the same reel.
- AI category sidecars are persisted next to the source archive so future reruns skip already-classified posts unless you force overwrite.
- Google Maps import is meant for [Google My Maps](https://support.google.com/mymaps/answer/3024836), which accepts CSV imports. After importing there, you can still view the map in Google Maps.
- On macOS, use `caffeinate` for long runs if you plan to leave the machine unattended:

```bash
caffeinate -dimsu ig-obsidian sync --config config.json
```

## Tests

```bash
python3 -m unittest discover -s tests
```
