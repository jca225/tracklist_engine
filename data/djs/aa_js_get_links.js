/*
  This script is meant to be run in a browser. Given a `1001tracklist` link to a dj 
  (i.e., https://1001tracklist.com/dj/{dj_name}/index.html), it will collect all of the 
  relevant metadata pertaining to each DJ set, including the link to the set, on the page.
  Note that in order to get the entire dataset, one must scroll all the way down the page
  so as to load in all of the sets from the backend (the backend does lazy loading). It took approximately
  60 minutes for 99 sets, which is about a set and a half per minute. Note this hybrid approach to 
  web-scraping is much better than the purely human or purely computer approaches I have explored.

*/

(function () {
  function parseViews(text) {
    if (!text) return null;
    text = text.trim().toLowerCase();
    if (text.endsWith("k")) return Math.round(parseFloat(text) * 1000);
    if (text.endsWith("m")) return Math.round(parseFloat(text) * 1000000);
    return Number(text.replace(/,/g, "")) || null;
  }

  function parseTracksCounts(text) {
    if (!text) return { ided_tracks: null, total_tracks: null };
    text = text.trim().toLowerCase();
    if (!text.includes("/")) return { ided_tracks: null, total_tracks: null };
    const [left, right] = text.split("/", 2).map(s => s.trim());
    const total = /^\d+$/.test(right) ? Number(right) : null;
    if (left === "all") return { ided_tracks: total, total_tracks: total };
    const ided = /^\d+$/.test(left) ? Number(left) : null;
    return { ided_tracks: ided, total_tracks: total };
  }

  const rows = Array.from(
    document.querySelectorAll("#kTZXcvbn .bItm.action.oItm, .bItm.action.oItm")
  );

  const base = location.origin;
  const items = rows.map(row => {
    const tracklist_id = row.dataset.id || row.id || "";

    const titleLink = row.querySelector(".bTitle a");
    const title = titleLink ? titleLink.textContent.trim() : "";
    const url = titleLink
      ? new URL(titleLink.getAttribute("href"), base).href
      : "";

    const dateEl = row.querySelector('[title="tracklist date"]');
    const date = dateEl ? dateEl.textContent.trim() : null;

    const creatorLink = row.querySelector(".tlUser a");
    const creator_name = creatorLink ? creatorLink.textContent.trim() : null;
    const creator_url = creatorLink
      ? new URL(creatorLink.getAttribute("href"), base).href
      : null;

    const viewsEl = row.querySelector(".badge.views");
    let views_text = viewsEl ? viewsEl.textContent.trim() : "";
    views_text = views_text.replace(/\s*views?/i, "").trim();
    const views = parseViews(views_text);

    const tracksEl = row.querySelector('[title="IDed tracks / total tracks"]');
    const tracks_text = tracksEl ? tracksEl.textContent.trim() : "";
    const { ided_tracks, total_tracks } = parseTracksCounts(tracks_text);

    const playTimeEl = row.querySelector('[title="play time"]');
    const play_time = playTimeEl ? playTimeEl.textContent.trim() : null;

    const likesEl = row.querySelector("div.likes");
    let likes = null;
    if (likesEl) {
      const lt = likesEl.textContent.trim();
      likes = /^\d+$/.test(lt) ? Number(lt) : null;
    }

    const stylesEl = row.querySelector('[title="musicstyle(s)"]');
    const styles = stylesEl ? stylesEl.textContent.trim() : null;

    return {
      tracklist_id,
      title,
      url,
      date,
      creator_name,
      creator_url,
      views,
      ided_tracks,
      total_tracks,
      play_time,
      likes,
      styles,
    };
  });

  // Pretty-print to console
  console.log("Extracted", items.length, "tracklists");
  console.log(items);

  // Try to copy JSON to clipboard (works in most devtools)
  const json = JSON.stringify(items, null, 2);
  try {
    if (typeof copy === "function") {
      copy(json);
      console.log("JSON copied to clipboard.");
    } else {
      console.log(
        "No copy() function; manually copy from console. JSON below:\n",
        json
      );
    }
  } catch (e) {
    console.log("Could not auto-copy. JSON below:\n", json);
  }

  return items;
})();
