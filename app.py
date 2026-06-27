import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import time
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.model_builder import load_saved_model
from src.preprocessing import preprocess_pil_image, prepare_preview_image
from src.config import ASSETS_DIR, REPORTS_DIR

st.set_page_config(
    page_title="DigitVision AI — Handwritten Digit Recognition",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ============================================================
   DESIGN TOKENS
   ============================================================ */
:root {
    --bg:          #050816;
    --bg2:         #0F172A;
    --card:        rgba(255,255,255,0.05);
    --card-hover:  rgba(255,255,255,0.08);
    --border:      rgba(255,255,255,0.08);
    --border-glow: rgba(124,58,237,0.5);
    --purple:      #7C3AED;
    --cyan:        #06B6D4;
    --highlight:   #A855F7;
    --success:     #22C55E;
    --text:        #F8FAFC;
    --text2:       #94A3B8;
    --text3:       #64748B;
    --grad:        linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%);
    --grad2:       linear-gradient(135deg, #A855F7 0%, #06B6D4 60%, #22C55E 100%);
    --shadow:      0 20px 60px rgba(0,0,0,0.6);
    --shadow-glow: 0 0 40px rgba(124,58,237,0.25);
}

/* ============================================================
   GLOBAL RESETS & BASE
   ============================================================ */
html { scroll-behavior: smooth; }

.stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    overflow-x: hidden;
}

/* animated background mesh */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* animated gradient blobs */
.stApp::after {
    content: '';
    position: fixed;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(124,58,237,0.12) 0%, transparent 70%);
    top: -100px;
    right: -100px;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: blobPulse 8s ease-in-out infinite alternate;
}

@keyframes blobPulse {
    0%   { transform: scale(1) translate(0,0); opacity: 0.8; }
    100% { transform: scale(1.3) translate(-50px,50px); opacity: 0.4; }
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }
.main .block-container { padding-top: 5.5rem !important; padding-bottom: 2rem !important; max-width: 1400px; }

/* scroll progress bar */
.scroll-progress {
    position: fixed; top: 0; left: 0; height: 3px;
    background: var(--grad); z-index: 9999;
    transition: width 0.1s linear;
}

/* ============================================================
   STICKY NAV
   ============================================================ */
.nav {
    position: fixed; top: 0; left: 0; right: 0; height: 68px;
    background: rgba(5,8,22,0.85);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-bottom: 1px solid var(--border);
    z-index: 1000;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 1px 0 rgba(255,255,255,0.04);
}
.nav-inner {
    width: 92%; max-width: 1380px;
    display: flex; align-items: center; justify-content: space-between;
}
.nav-logo {
    font-family: 'Space Grotesk', sans-serif; font-weight: 800;
    font-size: 1.35rem; color: #fff !important;
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--grad2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-decoration: none;
}
.nav-logo-icon { font-size: 1.5rem; -webkit-text-fill-color: initial; }
.nav-links { display: flex; gap: 2rem; }
.nav-links a {
    color: var(--text2) !important; text-decoration: none;
    font-size: 0.9rem; font-weight: 500;
    transition: color 0.2s; position: relative;
}
.nav-links a::after {
    content: ''; position: absolute; bottom: -4px; left: 0; right: 0;
    height: 2px; background: var(--grad); border-radius: 2px;
    transform: scaleX(0); transition: transform 0.3s;
}
.nav-links a:hover { color: #fff !important; }
.nav-links a:hover::after { transform: scaleX(1); }
.nav-ctas { display: flex; gap: 0.8rem; align-items: center; }
.btn-primary {
    background: var(--grad); color: #fff !important;
    padding: 0.5rem 1.3rem; border-radius: 10px;
    font-weight: 700; font-size: 0.88rem;
    text-decoration: none; transition: all 0.3s;
    box-shadow: 0 0 20px rgba(124,58,237,0.25);
    border: none;
}
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 0 30px rgba(124,58,237,0.5); }
.btn-ghost {
    border: 1px solid var(--border); color: var(--text2) !important;
    padding: 0.5rem 1.3rem; border-radius: 10px;
    font-weight: 600; font-size: 0.88rem;
    text-decoration: none; transition: all 0.3s;
    background: rgba(255,255,255,0.02);
}
.btn-ghost:hover { border-color: rgba(255,255,255,0.25); color: #fff !important; background: rgba(255,255,255,0.06); }

/* ============================================================
   HERO SECTION
   ============================================================ */
.hero {
    position: relative;
    min-height: 90vh;
    display: flex; align-items: center; justify-content: space-between;
    gap: 4rem; padding: 4rem 0 3rem;
    overflow: hidden;
}

/* animated floating digits */
.floating-digits {
    position: absolute; inset: 0; pointer-events: none; overflow: hidden;
    z-index: 0;
}
.float-digit {
    position: absolute;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 900; opacity: 0.04;
    color: var(--highlight);
    animation: floatDigit linear infinite;
    user-select: none;
}
@keyframes floatDigit {
    0%   { transform: translateY(110vh) rotate(-10deg); opacity: 0; }
    5%   { opacity: 0.06; }
    95%  { opacity: 0.04; }
    100% { transform: translateY(-10vh) rotate(10deg); opacity: 0; }
}

/* blobs behind hero */
.hero-blob-1 {
    position: absolute; width: 700px; height: 700px; border-radius: 50%;
    background: radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 70%);
    top: -200px; left: -150px; pointer-events: none; z-index: 0;
    animation: blobPulse 10s ease-in-out infinite alternate;
}
.hero-blob-2 {
    position: absolute; width: 500px; height: 500px; border-radius: 50%;
    background: radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%);
    bottom: -100px; right: -100px; pointer-events: none; z-index: 0;
    animation: blobPulse 12s ease-in-out infinite alternate-reverse;
}

.hero-left { flex: 1.1; position: relative; z-index: 1; }
.hero-right { flex: 0.9; display: flex; justify-content: center; align-items: center; position: relative; z-index: 1; }

.hero-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.3);
    border-radius: 30px; padding: 0.4rem 1rem;
    color: var(--highlight) !important; font-size: 0.82rem; font-weight: 700;
    letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 1.5rem;
    animation: badgePulse 3s ease-in-out infinite;
}
.hero-badge-dot {
    width: 6px; height: 6px; background: var(--success);
    border-radius: 50%; box-shadow: 0 0 8px var(--success);
    animation: dotPulse 1.5s ease-in-out infinite;
}
@keyframes badgePulse {
    0%,100% { box-shadow: 0 0 10px rgba(124,58,237,0.2); }
    50%      { box-shadow: 0 0 20px rgba(124,58,237,0.4); }
}
@keyframes dotPulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(0.7); }
}

.hero-headline {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 900; font-size: 3.8rem;
    line-height: 1.08; letter-spacing: -2px;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #fff 30%, #c4b5fd 65%, #67e8f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: gradShift 5s ease-in-out infinite alternate;
    background-size: 200% 200%;
}
@keyframes gradShift {
    0%   { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}
.hero-sub {
    color: var(--text2) !important; font-size: 1.15rem;
    line-height: 1.65; margin-bottom: 2.5rem; max-width: 520px;
}
.hero-actions { display: flex; gap: 1rem; flex-wrap: wrap; }
.btn-hero-primary {
    background: var(--grad); color: #fff !important;
    padding: 0.95rem 2.4rem; border-radius: 14px;
    font-weight: 800; font-size: 1rem; text-decoration: none;
    box-shadow: 0 4px 25px rgba(124,58,237,0.45);
    transition: all 0.3s; border: 1px solid rgba(255,255,255,0.1);
    position: relative; overflow: hidden;
}
.btn-hero-primary::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent);
    opacity: 0; transition: opacity 0.3s;
}
.btn-hero-primary:hover { transform: translateY(-3px); box-shadow: 0 8px 40px rgba(124,58,237,0.7); }
.btn-hero-primary:hover::before { opacity: 1; }
.btn-hero-secondary {
    border: 1px solid rgba(255,255,255,0.15); color: #fff !important;
    padding: 0.95rem 2.4rem; border-radius: 14px;
    font-weight: 700; font-size: 1rem; text-decoration: none;
    transition: all 0.3s; background: rgba(255,255,255,0.03);
}
.btn-hero-secondary:hover { background: rgba(255,255,255,0.08); transform: translateY(-3px); border-color: rgba(255,255,255,0.3); }

/* hero stat cards */
.hero-stats { display: flex; gap: 1rem; margin-top: 2.5rem; flex-wrap: wrap; }
.stat-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 0.9rem 1.4rem;
    text-align: center; transition: all 0.3s;
}
.stat-card:hover { border-color: rgba(124,58,237,0.4); transform: translateY(-2px); box-shadow: 0 0 25px rgba(124,58,237,0.12); }
.stat-val { font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 800; color: #fff !important; line-height: 1; }
.stat-val.cyan  { background: linear-gradient(135deg,#06B6D4,#7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stat-val.purple{ background: linear-gradient(135deg,#A855F7,#06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stat-lbl { font-size: 0.72rem; color: var(--text2); text-transform: uppercase; letter-spacing: 1px; margin-top: 0.25rem; }

/* hero neural illustration */
.neural-frame {
    width: 420px; height: 400px;
    background: radial-gradient(circle at 50% 40%, rgba(124,58,237,0.1) 0%, rgba(6,182,212,0.05) 60%, transparent 100%);
    border: 1px solid var(--border); border-radius: 30px;
    position: relative; overflow: hidden;
}
.neural-nodes {
    position: absolute; inset: 0;
    background-image:
        radial-gradient(circle at 20% 25%, var(--cyan)   3px, transparent 4px),
        radial-gradient(circle at 50% 15%, var(--purple) 4px, transparent 5px),
        radial-gradient(circle at 80% 25%, var(--cyan)   3px, transparent 4px),
        radial-gradient(circle at 15% 55%, var(--highlight) 4px, transparent 5px),
        radial-gradient(circle at 40% 50%, var(--cyan)   2px, transparent 3px),
        radial-gradient(circle at 60% 50%, var(--purple) 2px, transparent 3px),
        radial-gradient(circle at 85% 55%, var(--highlight) 4px, transparent 5px),
        radial-gradient(circle at 25% 80%, var(--cyan)   3px, transparent 4px),
        radial-gradient(circle at 50% 85%, var(--purple) 4px, transparent 5px),
        radial-gradient(circle at 75% 80%, var(--cyan)   3px, transparent 4px);
    opacity: 0.75;
    animation: nodePulse 4s ease-in-out infinite alternate;
}
@keyframes nodePulse {
    0%   { opacity: 0.5; }
    100% { opacity: 1; }
}
.neural-lines {
    position: absolute; inset: 0;
    background-image:
        linear-gradient(135deg, rgba(6,182,212,0.06) 0%, transparent 50%),
        linear-gradient(45deg, rgba(124,58,237,0.06) 0%, transparent 50%);
    animation: lineGlow 6s ease-in-out infinite alternate;
}
@keyframes lineGlow {
    0%   { opacity: 0.4; }
    100% { opacity: 1; }
}

/* floating cards inside hero graphic */
.hero-fc {
    position: absolute;
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px; padding: 1rem 1.2rem;
    box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    animation: heroFloat 5s ease-in-out infinite alternate;
}
.hero-fc.card-a { top: 20px; left: 10px; width: 190px; border-color: rgba(6,182,212,0.2); animation-delay: 0s; }
.hero-fc.card-b { bottom: 20px; right: 10px; width: 200px; border-color: rgba(124,58,237,0.2); animation-delay: 1.5s; }
@keyframes heroFloat {
    0%   { transform: translateY(-8px) rotate(-1deg); }
    100% { transform: translateY(8px) rotate(1deg); }
}
.fc-label { font-size: 0.65rem; color: var(--text2); text-transform: uppercase; letter-spacing: 1px; }
.fc-val   { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 800; color: #fff; }
.fc-sub   { font-size: 0.7rem; color: var(--cyan); font-weight: 600; margin-top: 0.15rem; }

/* ============================================================
   SECTION CHROME
   ============================================================ */
.section-tag {
    display: inline-block;
    background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.25);
    border-radius: 30px; padding: 0.3rem 0.9rem;
    color: var(--highlight) !important; font-size: 0.78rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem; font-weight: 800;
    color: #fff !important; margin-bottom: 0.6rem; line-height: 1.1;
}
.section-desc { color: var(--text2) !important; font-size: 1.05rem; line-height: 1.6; margin-bottom: 2.5rem; max-width: 600px; }
.section-center { text-align: center; }
.section-center .section-desc { margin-left: auto; margin-right: auto; }

/* ============================================================
   GLASS CARD BASE
   ============================================================ */
.gc {
    background: var(--card);
    backdrop-filter: blur(20px) saturate(150%);
    -webkit-backdrop-filter: blur(20px) saturate(150%);
    border: 1px solid var(--border);
    border-radius: 24px; padding: 2rem;
    box-shadow: var(--shadow); margin-bottom: 1.5rem;
    transition: border-color 0.3s, box-shadow 0.3s, transform 0.3s;
    position: relative; overflow: hidden;
}
.gc::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, transparent 50%);
    pointer-events: none;
}
.gc:hover {
    border-color: rgba(124,58,237,0.35);
    box-shadow: var(--shadow), var(--shadow-glow);
    transform: translateY(-3px);
}
.gc-title {
    display: flex; align-items: center; gap: 0.6rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem; font-weight: 700;
    color: #fff !important; margin-bottom: 1.2rem;
}
.gc-title-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--cyan); box-shadow: 0 0 12px var(--cyan);
    flex-shrink: 0;
}

/* ============================================================
   SEGMENTED CONTROL (INPUT MODE SELECTOR)
   ============================================================ */
div[role="radiogroup"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 5px !important;
    display: flex !important;
    gap: 6px !important;
    margin-bottom: 1.8rem !important;
}
div[role="radiogroup"] > label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    color: var(--text2) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    cursor: pointer !important;
    transition: all 0.25s !important;
    flex: 1 !important; text-align: center !important;
    display: inline-flex !important; align-items: center !important; justify-content: center !important;
}
div[role="radiogroup"] > label:hover { color: #fff !important; background: rgba(255,255,255,0.05) !important; }
div[role="radiogroup"] > label div[role="presentation"],
div[role="radiogroup"] > label input[type="radio"] { display: none !important; width: 0 !important; opacity: 0 !important; }
div[role="radiogroup"] > label:has(input[checked]) {
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(6,182,212,0.2)) !important;
    border-color: rgba(124,58,237,0.4) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 18px rgba(124,58,237,0.15) !important;
}

/* ============================================================
   DEMO CANVAS WRAPPER
   ============================================================ */
.canvas-wrapper {
    border: 2px solid rgba(6,182,212,0.3);
    border-radius: 20px; overflow: hidden;
    background: #fff;
    box-shadow: 0 0 30px rgba(6,182,212,0.15), 0 0 0 1px rgba(6,182,212,0.1);
    display: inline-block;
    transition: box-shadow 0.3s;
}
.canvas-wrapper:hover { box-shadow: 0 0 45px rgba(124,58,237,0.35); }

/* ============================================================
   PREDICTION RING
   ============================================================ */
.pred-ring {
    width: 160px; height: 160px; border-radius: 50%; margin: 0 auto 1rem;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: radial-gradient(circle, rgba(124,58,237,0.18) 0%, rgba(6,182,212,0.04) 100%);
    border: 3px solid var(--cyan);
    box-shadow: 0 0 30px rgba(6,182,212,0.4), inset 0 0 20px rgba(6,182,212,0.1);
    animation: ringPulse 3s ease-in-out infinite alternate;
}
@keyframes ringPulse {
    0%   { box-shadow: 0 0 20px rgba(6,182,212,0.35), inset 0 0 12px rgba(6,182,212,0.1); border-color: var(--cyan); }
    100% { box-shadow: 0 0 45px rgba(168,85,247,0.55), inset 0 0 25px rgba(168,85,247,0.2); border-color: var(--highlight); }
}
.pred-ring-lbl { font-size: 0.62rem; color: var(--text2); text-transform: uppercase; letter-spacing: 1px; }
.pred-ring-val { font-family: 'Space Grotesk', sans-serif; font-size: 5rem; font-weight: 900; color: #fff; line-height: 1; text-shadow: 0 0 20px rgba(255,255,255,0.5); }

/* ============================================================
   PROBABILITY BARS
   ============================================================ */
.prob-bar-row { margin-bottom: 0.55rem; }
.prob-bar-top { display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 0.15rem; color: var(--text) !important; }
.prob-bar-bg { background: rgba(255,255,255,0.05); border-radius: 6px; height: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.04); }
.prob-bar-fill { height: 100%; border-radius: 6px; transition: width 0.5s cubic-bezier(0.4,0,0.2,1); }

/* ============================================================
   METRICS DASHBOARD
   ============================================================ */
.metrics-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-top: 1.5rem; }
.metric-card {
    background: rgba(255,255,255,0.03); border: 1px solid var(--border);
    border-radius: 20px; padding: 1.5rem 1rem; text-align: center;
    transition: all 0.3s; position: relative; overflow: hidden;
}
.metric-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: var(--grad); opacity: 0; transition: opacity 0.3s;
}
.metric-card:hover { border-color: rgba(124,58,237,0.4); background: rgba(255,255,255,0.05); transform: translateY(-4px); box-shadow: var(--shadow-glow); }
.metric-card:hover::after { opacity: 1; }
.metric-val  { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 900; background: var(--grad2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-lbl  { font-size: 0.72rem; color: var(--text2); text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; font-weight: 600; }
.metric-icon { font-size: 1.6rem; margin-bottom: 0.5rem; display: block; }

/* ============================================================
   FEATURES GRID
   ============================================================ */
.feat-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap: 1.2rem; margin-top: 2rem; }
.feat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 20px; padding: 1.8rem;
    transition: all 0.3s; position: relative; overflow: hidden;
}
.feat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: var(--grad); opacity: 0; transition: opacity 0.3s;
}
.feat-card:hover { transform: translateY(-6px); border-color: rgba(124,58,237,0.4); box-shadow: 0 8px 35px rgba(124,58,237,0.12); }
.feat-card:hover::before { opacity: 1; }
.feat-icon { font-size: 2.4rem; margin-bottom: 1rem; display: inline-block; }
.feat-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; color: #fff !important; margin-bottom: 0.5rem; }
.feat-desc  { color: var(--text2) !important; font-size: 0.88rem; line-height: 1.5; }

/* ============================================================
   HOW IT WORKS TIMELINE
   ============================================================ */
.timeline { display: flex; align-items: stretch; gap: 0; margin-top: 2rem; flex-wrap: wrap; }
.timeline-step {
    flex: 1; min-width: 150px;
    background: var(--card); border: 1px solid var(--border); border-radius: 18px;
    padding: 1.5rem 1rem; text-align: center; transition: all 0.3s;
}
.timeline-step:hover { border-color: rgba(6,182,212,0.4); transform: translateY(-4px); }
.timeline-arrow { display: flex; align-items: center; color: var(--cyan); font-size: 1.6rem; padding: 0 0.4rem; animation: arrowPulse 1.5s ease-in-out infinite; }
@keyframes arrowPulse { 0%,100%{opacity:0.5;transform:scale(0.9)} 50%{opacity:1;transform:scale(1.1)} }
.tl-num { font-family: 'Space Grotesk', sans-serif; font-size: 0.7rem; font-weight: 700; color: var(--cyan); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem; }
.tl-icon { font-size: 2rem; margin-bottom: 0.5rem; display: block; }
.tl-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 700; color: #fff !important; margin-bottom: 0.4rem; }
.tl-desc  { font-size: 0.75rem; color: var(--text2); line-height: 1.4; }

/* ============================================================
   CNN ARCHITECTURE
   ============================================================ */
.cnn-layers { display: flex; flex-direction: column; gap: 0.5rem; margin-top: 1rem; }
.cnn-row {
    display: flex; align-items: center; gap: 1rem;
    background: rgba(255,255,255,0.025); border: 1px solid var(--border);
    border-radius: 14px; padding: 0.75rem 1rem;
    transition: background 0.2s, border-color 0.2s;
}
.cnn-row:hover { background: rgba(255,255,255,0.05); border-color: rgba(124,58,237,0.3); }
.cnn-badge {
    padding: 0.25rem 0.75rem; border-radius: 8px; font-size: 0.72rem;
    font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;
    color: #fff; min-width: 90px; text-align: center; flex-shrink: 0;
}
.badge-in  { background: linear-gradient(135deg,#3b82f6,#1e40af); }
.badge-cv  { background: linear-gradient(135deg,#10b981,#059669); }
.badge-bn  { background: linear-gradient(135deg,#8b5cf6,#6d28d9); }
.badge-mp  { background: linear-gradient(135deg,#f59e0b,#d97706); }
.badge-do  { background: linear-gradient(135deg,#ef4444,#b91c1c); }
.badge-fl  { background: linear-gradient(135deg,#6366f1,#4338ca); }
.badge-fc  { background: linear-gradient(135deg,#ec4899,#be185d); }
.badge-sm  { background: linear-gradient(135deg,#f97316,#ea580c); }
.cnn-arrow { text-align: center; color: rgba(6,182,212,0.6); font-size: 1rem; padding: 0; margin: -0.12rem 0; }
.cnn-info  { font-size: 0.88rem; color: var(--text) !important; }
.cnn-info strong { color: #fff !important; }
.cnn-dim   { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--cyan); margin-left: auto; white-space: nowrap; }

/* model params table */
.params-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
.params-table tr { border-bottom: 1px solid var(--border); }
.params-table tr:last-child { border-bottom: none; }
.params-table td { padding: 0.85rem 0.5rem; font-size: 0.9rem; }
.params-table td:first-child { color: var(--text2) !important; font-weight: 500; }
.params-table td:last-child  { color: #fff !important; font-weight: 700; text-align: right; }

/* ============================================================
   TECH STACK
   ============================================================ */
.tech-cloud { display: flex; flex-wrap: wrap; gap: 0.9rem; justify-content: center; margin-top: 2rem; }
.tech-pill {
    display: flex; align-items: center; gap: 0.5rem;
    background: rgba(255,255,255,0.03); border: 1px solid var(--border);
    border-radius: 50px; padding: 0.6rem 1.4rem;
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.9rem;
    color: var(--text) !important; transition: all 0.3s; cursor: default;
}
.tech-pill:hover { border-color: var(--cyan); color: #fff !important; box-shadow: 0 0 20px rgba(6,182,212,0.2); transform: translateY(-3px); }
.tech-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }

/* ============================================================
   PROJECT HIGHLIGHTS
   ============================================================ */
.hl-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(250px,1fr)); gap: 1.2rem; margin-top: 2rem; }
.hl-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 20px; padding: 1.8rem; transition: all 0.3s;
    position: relative; overflow: hidden;
}
.hl-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 20px 20px 0 0; opacity: 0; transition: opacity 0.3s;
}
.hl-card.hl-green::after { background: var(--success); }
.hl-card.hl-blue::after  { background: var(--cyan); }
.hl-card.hl-purple::after{ background: var(--highlight); }
.hl-card.hl-white::after { background: linear-gradient(135deg,#fff,rgba(255,255,255,0.3)); }
.hl-card.hl-orange::after{ background: linear-gradient(135deg,#f97316,#ef4444); }
.hl-card:hover { transform: translateY(-5px); border-color: rgba(124,58,237,0.35); }
.hl-card:hover::after { opacity: 1; }
.hl-icon  { font-size: 2rem; margin-bottom: 0.8rem; display: block; }
.hl-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.4rem; color: #fff !important; }
.hl-desc  { font-size: 0.85rem; color: var(--text2) !important; line-height: 1.5; }

/* ============================================================
   ABOUT PROJECT
   ============================================================ */
.about-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 2rem; }
.about-item {
    background: rgba(255,255,255,0.02); border: 1px solid var(--border);
    border-radius: 18px; padding: 1.5rem; transition: all 0.3s;
}
.about-item:hover { border-color: rgba(124,58,237,0.3); background: rgba(255,255,255,0.04); }
.about-item-icon { font-size: 1.8rem; margin-bottom: 0.7rem; display: block; }
.about-item-title { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; color: #fff !important; margin-bottom: 0.5rem; }
.about-item-text  { color: var(--text2) !important; font-size: 0.88rem; line-height: 1.55; }

/* ============================================================
   DEVELOPER PROFILE
   ============================================================ */
.dev-wrapper {
    display: flex; gap: 3rem; align-items: flex-start; flex-wrap: wrap;
}
.dev-left {
    flex: 0 0 260px; display: flex; flex-direction: column; align-items: center; text-align: center;
}
.dev-avatar {
    width: 150px; height: 150px; border-radius: 50%;
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(6,182,212,0.35));
    border: 3px solid rgba(255,255,255,0.08);
    display: flex; align-items: center; justify-content: center;
    font-size: 4rem; margin-bottom: 1rem;
    box-shadow: 0 0 40px rgba(124,58,237,0.25), 0 0 80px rgba(6,182,212,0.1);
    animation: devFloat 5s ease-in-out infinite alternate;
}
@keyframes devFloat {
    0%   { transform: translateY(-5px); }
    100% { transform: translateY(5px); }
}
.dev-name  { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 800; color: #fff !important; margin-bottom: 0.3rem; }
.dev-title { color: var(--cyan) !important; font-size: 0.9rem; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 1rem; }
.dev-badges { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1.2rem; }
.dev-badge  {
    display: inline-block; border-radius: 20px;
    padding: 0.3rem 0.8rem; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;
}
.badge-avail  { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); color: var(--success) !important; }
.badge-oss    { background: rgba(6,182,212,0.12); border: 1px solid rgba(6,182,212,0.3); color: var(--cyan) !important; }
.badge-ai     { background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.3); color: var(--highlight) !important; }

.dev-right { flex: 1; }
.dev-bio   { color: var(--text2) !important; font-size: 0.95rem; line-height: 1.65; margin-bottom: 1.5rem; }
.skills-flex { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem; }
.skill-tag {
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.25rem 0.7rem;
    font-size: 0.78rem; font-weight: 600; color: var(--text2) !important;
    transition: all 0.2s;
}
.skill-tag:hover { border-color: var(--purple); color: #fff !important; }

.dev-socials { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.social-btn {
    display: flex; align-items: center; gap: 0.5rem;
    border: 1px solid var(--border); border-radius: 10px;
    padding: 0.5rem 1rem; font-size: 0.85rem; font-weight: 600;
    color: var(--text2) !important; text-decoration: none;
    background: rgba(255,255,255,0.02); transition: all 0.2s;
}
.social-btn:hover { border-color: rgba(255,255,255,0.2); color: #fff !important; transform: translateY(-2px); }

.dev-action-btns { display: flex; gap: 0.8rem; flex-wrap: wrap; }
.dev-btn-primary {
    background: var(--grad); color: #fff !important;
    padding: 0.65rem 1.4rem; border-radius: 10px;
    font-weight: 700; font-size: 0.9rem; text-decoration: none;
    transition: all 0.3s; box-shadow: 0 0 20px rgba(124,58,237,0.25);
}
.dev-btn-primary:hover { transform: translateY(-2px); box-shadow: 0 0 30px rgba(124,58,237,0.5); }
.dev-btn-secondary {
    border: 1px solid var(--border); color: var(--text2) !important;
    padding: 0.65rem 1.4rem; border-radius: 10px;
    font-weight: 600; font-size: 0.9rem; text-decoration: none;
    transition: all 0.3s; background: rgba(255,255,255,0.02);
}
.dev-btn-secondary:hover { border-color: rgba(255,255,255,0.2); color: #fff !important; }

/* ============================================================
   FOOTER
   ============================================================ */
.footer {
    background: var(--bg2); border-top: 1px solid var(--border);
    border-radius: 28px 28px 0 0; padding: 4rem 3rem 2rem;
    margin-top: 4rem; position: relative; overflow: hidden;
}
.footer::before {
    content: ''; position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 300px; height: 1px; background: var(--grad);
}
.footer-grid { display: grid; grid-template-columns: 1.5fr 1fr 1fr 1.5fr; gap: 3rem; margin-bottom: 3rem; }
.footer-logo { font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 1.4rem; background: var(--grad2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.8rem; }
.footer-tagline { color: var(--text2); font-size: 0.88rem; line-height: 1.5; }
.footer-col-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #fff !important; margin-bottom: 1.2rem; }
.footer-link { display: block; color: var(--text3) !important; text-decoration: none; font-size: 0.88rem; margin-bottom: 0.6rem; transition: color 0.2s; }
.footer-link:hover { color: #fff !important; }
.footer-contact-item { display: flex; flex-direction: column; gap: 0.1rem; margin-bottom: 0.7rem; }
.footer-contact-lbl { font-size: 0.72rem; color: var(--text3); text-transform: uppercase; letter-spacing: 0.5px; }
.footer-contact-val { font-size: 0.85rem; color: var(--text2) !important; text-decoration: none; }
.footer-contact-val:hover { color: #fff !important; }
.footer-bottom { border-top: 1px solid var(--border); padding-top: 2rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; }
.footer-copy { font-size: 0.82rem; color: var(--text3); }
.footer-built { font-size: 0.82rem; color: var(--text3); }

/* ============================================================
   PREDICTION HISTORY
   ============================================================ */
.pred-history { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
.ph-chip {
    background: rgba(255,255,255,0.05); border: 1px solid var(--border);
    border-radius: 30px; padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    color: var(--text2) !important; transition: all 0.2s;
}
.ph-chip-active {
    background: rgba(6,182,212,0.12); border-color: rgba(6,182,212,0.3);
    color: var(--cyan) !important;
}

/* ============================================================
   LOADING SPINNER
   ============================================================ */
.spin-wrapper { display: flex; flex-direction: column; align-items: center; gap: 1rem; padding: 2rem; }
.spinner {
    width: 48px; height: 48px; border-radius: 50%;
    border: 3px solid rgba(255,255,255,0.08);
    border-top-color: var(--cyan);
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ============================================================
   UTILITY
   ============================================================ */
.divider { height: 1px; background: var(--border); margin: 3rem 0; }
.mt-1 { margin-top: 1rem !important; }
.mt-2 { margin-top: 2rem !important; }
.text-cyan { color: var(--cyan) !important; }
.text-purple { color: var(--highlight) !important; }
.text-white { color: #fff !important; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* Force all Streamlit markdown text to be readable */
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] span,
.stApp [data-testid="stMarkdownContainer"] li { color: var(--text2) !important; }
.stApp [data-testid="stMarkdownContainer"] strong { color: var(--text) !important; }
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3 { color: #fff !important; font-family: 'Space Grotesk', sans-serif !important; }

/* Checkbox & info boxes */
.stCheckbox label { color: var(--text2) !important; }
.stInfo  { background: rgba(6,182,212,0.1)  !important; border: 1px solid rgba(6,182,212,0.25) !important; border-radius: 12px !important; color: var(--text2) !important; }
.stAlert { background: rgba(124,58,237,0.1) !important; border: 1px solid rgba(124,58,237,0.25) !important; border-radius: 12px !important; }

/* File uploader */
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(6,182,212,0.25) !important;
    border-radius: 16px !important;
}
/* Buttons (sample selector) */
div.stButton > button {
    background: rgba(255,255,255,0.04); color: #fff;
    border: 1px solid var(--border); border-radius: 12px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700; font-size: 1.3rem;
    height: 3rem; width: 100%;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: rgba(124,58,237,0.15) !important;
    border-color: var(--purple) !important;
    color: var(--highlight) !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.3) !important;
    transform: translateY(-3px);
}

/* ============================================================
   SCROLL PROGRESS JS INJECTION
   ============================================================ */
</style>

<script>
// Scroll progress bar
window.addEventListener('scroll', () => {
    const bar = document.getElementById('scroll-bar');
    if (!bar) return;
    const pct = (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
    bar.style.width = pct + '%';
});

// Dynamic renderer for the prediction panel using pure DOM creation methods (no innerText, textContent, or innerHTML)
window.renderPrediction = (prediction, confidence, inf_ms, version, probs) => {
    const container = document.getElementById('prediction-panel-container');
    if (!container) return;
    
    // Clear previous predictions
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }
    
    // Create elements
    const card = document.createElement('div');
    card.className = 'gc';
    
    const title = document.createElement('div');
    title.className = 'gc-title';
    const dot = document.createElement('div');
    dot.className = 'gc-title-dot';
    title.appendChild(dot);
    title.appendChild(document.createTextNode('AI Classification Result'));
    card.appendChild(title);
    
    const layout = document.createElement('div');
    layout.style.display = 'flex';
    layout.style.gap = '2rem';
    layout.style.alignItems = 'center';
    layout.style.flexWrap = 'wrap';
    
    // Left column (Ring & Stats)
    const leftCol = document.createElement('div');
    leftCol.style.textAlign = 'center';
    leftCol.style.flex = '0 0 auto';
    
    const ring = document.createElement('div');
    ring.className = 'pred-ring';
    
    const ringLbl = document.createElement('div');
    ringLbl.className = 'pred-ring-lbl';
    ringLbl.appendChild(document.createTextNode('Digit'));
    
    const ringVal = document.createElement('div');
    ringVal.className = 'pred-ring-val';
    ringVal.appendChild(document.createTextNode(prediction.toString()));
    
    ring.appendChild(ringLbl);
    ring.appendChild(ringVal);
    leftCol.appendChild(ring);
    
    const confDiv = document.createElement('div');
    confDiv.style.color = '#fff';
    confDiv.style.fontWeight = '700';
    confDiv.style.fontSize = '1rem';
    confDiv.appendChild(document.createTextNode('Confidence: '));
    
    const confSpan = document.createElement('span');
    confSpan.style.color = 'var(--cyan)';
    confSpan.appendChild(document.createTextNode(confidence.toFixed(2) + '%'));
    confDiv.appendChild(confSpan);
    leftCol.appendChild(confDiv);
    
    const statsDiv = document.createElement('div');
    statsDiv.style.fontSize = '0.75rem';
    statsDiv.style.color = 'var(--text2)';
    statsDiv.style.marginTop = '0.5rem';
    statsDiv.style.lineHeight = '1.5';
    
    statsDiv.appendChild(document.createTextNode('Inference: '));
    const infSpan = document.createElement('span');
    infSpan.className = 'mono';
    infSpan.style.color = 'var(--cyan)';
    infSpan.appendChild(document.createTextNode(inf_ms.toFixed(1) + 'ms'));
    statsDiv.appendChild(infSpan);
    
    statsDiv.appendChild(document.createElement('br'));
    
    statsDiv.appendChild(document.createTextNode('Version: '));
    const verSpan = document.createElement('span');
    verSpan.className = 'mono';
    verSpan.appendChild(document.createTextNode(version));
    statsDiv.appendChild(verSpan);
    
    leftCol.appendChild(statsDiv);
    layout.appendChild(leftCol);
    
    // Right column (Probability Distribution)
    const rightCol = document.createElement('div');
    rightCol.style.flex = '1';
    
    const distTitle = document.createElement('div');
    distTitle.style.fontSize = '0.85rem';
    distTitle.style.fontWeight = '700';
    distTitle.style.color = '#fff';
    distTitle.style.marginBottom = '0.8rem';
    distTitle.appendChild(document.createTextNode('All-class Probability Distribution'));
    rightCol.appendChild(distTitle);
    
    for (let d = 0; d < 10; d++) {
        const p = probs[d];
        
        const row = document.createElement('div');
        row.className = 'prob-bar-row';
        
        const top = document.createElement('div');
        top.className = 'prob-bar-top';
        
        const labelSpan = document.createElement('span');
        if (d === prediction) {
            labelSpan.appendChild(document.createTextNode('★ Digit ' + d));
        } else {
            labelSpan.appendChild(document.createTextNode('Digit ' + d));
        }
        
        const pctSpan = document.createElement('span');
        pctSpan.style.color = d === prediction ? 'var(--cyan)' : 'var(--text2)';
        pctSpan.appendChild(document.createTextNode((p * 100).toFixed(2) + '%'));
        
        top.appendChild(labelSpan);
        top.appendChild(pctSpan);
        
        const barBg = document.createElement('div');
        barBg.className = 'prob-bar-bg';
        
        const barFill = document.createElement('div');
        barFill.className = 'prob-bar-fill';
        barFill.style.width = '0%';
        barFill.style.background = d === prediction ? 'linear-gradient(90deg,#7C3AED,#06B6D4)' : 'rgba(255,255,255,0.12)';
        
        barBg.appendChild(barFill);
        row.appendChild(top);
        row.appendChild(barBg);
        rightCol.appendChild(row);
        
        // Animate the bar width fill using requestAnimationFrame or setTimeout
        (function(fillElem, targetWidth) {
            setTimeout(() => {
                fillElem.style.width = targetWidth;
            }, 50);
        })(barFill, (p * 100).toFixed(2) + '%');
    }
    
    layout.appendChild(rightCol);
    card.appendChild(layout);
    container.appendChild(card);
};
</script>

<div id="scroll-bar" class="scroll-progress" style="width:0%;"></div>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STICKY NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
  <div class="nav-inner">
    <a href="#home" class="nav-logo">
      <span class="nav-logo-icon">👁️</span>DigitVision AI
    </a>
    <div class="nav-links">
      <a href="#home">Home</a>
      <a href="#demo">Demo</a>
      <a href="#features">Features</a>
      <a href="#performance">Performance</a>
      <a href="#technology">Technology</a>
      <a href="#about">About</a>
    </div>
    <div class="nav-ctas">
      <a href="#demo" class="btn-primary">Try Live Demo</a>
      <a href="https://github.com/muhammadabdullah-devpk" target="_blank" class="btn-ghost">GitHub</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
float_digits_html = "".join([
    f'<span class="float-digit" style="left:{left}%;font-size:{size}rem;animation-duration:{dur}s;animation-delay:{delay}s;">{digit}</span>'
    for left,size,dur,delay,digit in [
        (5,4,18,0,'3'),(15,6,22,3,'7'),(25,3,15,1,'1'),(35,5,20,5,'5'),
        (50,7,25,2,'9'),(60,4,17,4,'0'),(70,5,21,1,'2'),(80,3,16,6,'4'),
        (90,6,23,3,'8'),(12,4,19,7,'6'),(45,3,14,2,'3'),(75,5,20,5,'7'),
    ]
])

st.markdown(f"""
<div class="hero" id="home">
  <div class="floating-digits">{float_digits_html}</div>
  <div class="hero-blob-1"></div>
  <div class="hero-blob-2"></div>

  <div class="hero-left">
    <div class="hero-badge"><span class="hero-badge-dot"></span>Live · 99.40% Accuracy on MNIST</div>
    <h1 class="hero-headline">Transform Handwritten<br/>Digits into Intelligent<br/>Predictions</h1>
    <p class="hero-sub">
      An advanced AI platform built on a Deep Convolutional Neural Network delivering
      state-of-the-art recognition with lightning-fast real-time inference on the MNIST dataset.
    </p>
    <div class="hero-actions">
      <a href="#demo" class="btn-hero-primary">🚀 Try Live Demo</a>
      <a href="https://github.com/muhammadabdullah-devpk" target="_blank" class="btn-hero-secondary">View GitHub</a>
    </div>
    <div class="hero-stats">
      <div class="stat-card">
        <div class="stat-val cyan">99.40%</div>
        <div class="stat-lbl">Test Accuracy</div>
      </div>
      <div class="stat-card">
        <div class="stat-val purple">1.0000</div>
        <div class="stat-lbl">ROC-AUC</div>
      </div>
      <div class="stat-card">
        <div class="stat-val cyan">70K</div>
        <div class="stat-lbl">Dataset Images</div>
      </div>
      <div class="stat-card">
        <div class="stat-val purple">~0.8ms</div>
        <div class="stat-lbl">Inference Time</div>
      </div>
    </div>
  </div>

  <div class="hero-right">
    <div class="neural-frame">
      <div class="neural-nodes"></div>
      <div class="neural-lines"></div>
      <div class="hero-fc card-a">
        <div class="fc-label">CNN Input</div>
        <div class="fc-val">28×28×1</div>
        <div class="fc-sub">Grayscale Normalised</div>
      </div>
      <div class="hero-fc card-b">
        <div class="fc-label">Prediction Output</div>
        <div class="fc-val" style="font-size:2.5rem;">Digit 7</div>
        <div class="fc-sub">Confidence: 99.67%</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div id="demo"></div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE DEMO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-center" style="margin-bottom:2rem;">
  <div class="section-tag">Interactive</div>
  <div class="section-title">👁️ Live Digit Recognition Demo</div>
  <div class="section-desc">
    Test the CNN model in real time. Draw a digit, upload an image, or pick a sample from the MNIST dataset.
  </div>
</div>
""", unsafe_allow_html=True)

input_mode = st.radio(
    "Input Method",
    ["✏️ Live Drawing Pad", "📂 Upload Image File", "⚡ Quick MNIST Samples"],
    horizontal=True, label_visibility="collapsed",
)

if "selected_sample"    not in st.session_state: st.session_state.selected_sample = None
if "pred_history"       not in st.session_state: st.session_state.pred_history    = []

demo_l, demo_r = st.columns([1, 1.3], gap="large")
active_image   = None
source_title   = ""

with demo_l:
    # ── Mode 1 : Drawing pad ──────────────────────────────────────────────
    if input_mode == "✏️ Live Drawing Pad":
        st.markdown('<div class="gc">', unsafe_allow_html=True)
        st.markdown('<div class="gc-title"><div class="gc-title-dot"></div>Live Drawing Canvas</div>', unsafe_allow_html=True)
        st.markdown("Draw any digit (0-9) below — predictions update automatically:", unsafe_allow_html=False)
        _, c2, _ = st.columns([1, 3.5, 1])
        with c2:
            st.markdown('<div class="canvas-wrapper">', unsafe_allow_html=True)
            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,1)",
                stroke_width=22, stroke_color="#000000",
                background_color="#ffffff",
                update_streamlit=True,
                height=285, width=285,
                drawing_mode="freedraw", key="draw_canvas",
            )
            st.markdown('</div>', unsafe_allow_html=True)
        st.caption("✏️  Draw clearly in the centre. Use toolbar (bottom of canvas) to undo or clear.")
        if canvas_result.image_data is not None:
            ci = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA").convert("L")
            if np.min(np.array(ci)) < 230:
                active_image = ci; source_title = "Drawing Canvas"
            else:
                st.info("Start drawing above to trigger AI inference.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Mode 2 : Upload ───────────────────────────────────────────────────
    elif input_mode == "📂 Upload Image File":
        st.markdown('<div class="gc">', unsafe_allow_html=True)
        st.markdown('<div class="gc-title"><div class="gc-title-dot"></div>Upload Digit Image</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;padding:1.8rem;border:2px dashed rgba(6,182,212,0.25);border-radius:18px;background:rgba(6,182,212,0.015);margin-bottom:1.2rem;">
          <span style="font-size:2.8rem;">📥</span>
          <h3 style="margin:0.7rem 0 0.3rem;color:#fff;font-size:1.25rem;">Drag & Drop or Browse</h3>
          <p style="margin:0;font-size:0.85rem;color:var(--text2);">PNG · JPG · JPEG — black digit on white background recommended.</p>
        </div>
        """, unsafe_allow_html=True)
        uf = st.file_uploader("", type=["png","jpg","jpeg"], key="up_tab", label_visibility="collapsed")
        if uf:
            try: active_image = Image.open(uf).convert("L"); source_title = "Uploaded File"
            except Exception as e: st.error(f"Error: {e}")
        else:
            st.info("Upload a handwritten digit image to begin.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Mode 3 : MNIST Samples ────────────────────────────────────────────
    else:
        st.markdown('<div class="gc">', unsafe_allow_html=True)
        st.markdown('<div class="gc-title"><div class="gc-title-dot"></div>Quick MNIST Samples</div>', unsafe_allow_html=True)
        st.markdown("Click any digit — loads a real MNIST test image:", unsafe_allow_html=False)
        cols = st.columns(10)
        for i in range(10):
            with cols[i]:
                if st.button(str(i), key=f"s{i}"):
                    st.session_state.selected_sample = i; st.rerun()
        if st.session_state.selected_sample is not None:
            sp = ASSETS_DIR / f"sample_{st.session_state.selected_sample}.png"
            if sp.exists():
                try: active_image = Image.open(sp).convert("L"); source_title = f"MNIST Sample — Digit {st.session_state.selected_sample}"
                except Exception as e: st.error(str(e))
            else: st.error(f"Sample not found: {sp.name}")
        else:
            st.info("Select a digit button to load the MNIST sample.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Preprocessing Preview ─────────────────────────────────────────────
    if active_image is not None:
        # Run prediction early
        with st.spinner(""):
            try:
                t0 = time.perf_counter()
                model = load_saved_model()
                probs = model.predict(preprocess_pil_image(active_image), verbose=0)[0]
                inf_ms = (time.perf_counter() - t0) * 1000
                prediction = int(np.argmax(probs))
                confidence = float(probs[prediction]) * 100
            except FileNotFoundError:
                st.error("Saved model not found — run `python train.py` first.")
                probs = None
                prediction = None
                confidence = 0.0
                inf_ms = 0
            except Exception as e:
                st.error(f"Inference error: {e}")
                probs = None
                prediction = None
                confidence = 0.0
                inf_ms = 0

        # Display sequential Pipeline Preview (Original -> Preprocessed -> Prediction -> Confidence)
        if probs is not None:
            st.markdown('<div class="gc" style="margin-top:0;">', unsafe_allow_html=True)
            st.markdown(f'<div class="gc-title"><div class="gc-title-dot"></div>Pipeline Preview · {source_title}</div>', unsafe_allow_html=True)
            
            # Center vertical flow with arrows
            st.markdown("<div style='display:flex; flex-direction:column; align-items:center; gap:0.6rem; text-align:center;'>", unsafe_allow_html=True)
            
            # 1. Original Image
            st.markdown("<div style='font-size:0.85rem; font-weight:700; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px;'>Original Image</div>", unsafe_allow_html=True)
            st.image(active_image, width=140)
            
            # Arrow
            st.markdown("<div style='color:var(--cyan); font-size:1.8rem; font-weight:900; line-height:1; margin:0.2rem 0;'>↓</div>", unsafe_allow_html=True)
            
            # 2. Preprocessed Image
            st.markdown("<div style='font-size:0.85rem; font-weight:700; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px;'>Preprocessed Image (28×28)</div>", unsafe_allow_html=True)
            st.image(prepare_preview_image(active_image), width=140, clamp=True)
            
            # Arrow
            st.markdown("<div style='color:var(--cyan); font-size:1.8rem; font-weight:900; line-height:1; margin:0.2rem 0;'>↓</div>", unsafe_allow_html=True)
            
            # 3. Prediction
            st.markdown("<div style='font-size:0.85rem; font-weight:700; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px;'>Prediction</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif; font-size:2.8rem; font-weight:900; color:#fff; line-height:1;'>Digit {prediction}</div>", unsafe_allow_html=True)
            
            # Arrow
            st.markdown("<div style='color:var(--cyan); font-size:1.8rem; font-weight:900; line-height:1; margin:0.2rem 0;'>↓</div>", unsafe_allow_html=True)
            
            # 4. Confidence
            st.markdown("<div style='font-size:0.85rem; font-weight:700; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px;'>Confidence</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.6rem; font-weight:800; color:var(--success); line-height:1;'>{confidence:.2f}%</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT COLUMN — Results ─────────────────────────────────────────────────
with demo_r:
    if active_image is None:
        st.markdown('<div id="prediction-panel-container"><div class="gc" style="text-align:center;padding:5rem 2rem;"><span style="font-size:4rem;">🧠</span><h3 style="color:#fff;margin:1rem 0 0.5rem;">Ready for Inference</h3><p style="color:var(--text2);">Draw a digit, upload an image, or pick a sample on the left.</p></div></div>', unsafe_allow_html=True)
    else:
        # Create persistent container empty on python side, and trigger the JS rendering immediately
        st.markdown('<div id="prediction-panel-container"></div>', unsafe_allow_html=True)
        if probs is not None:
            st.markdown(f"""
            <script>
            (function() {{
                const checkAndRender = () => {{
                    if (window.renderPrediction) {{
                        window.renderPrediction({prediction}, {confidence:.2f}, {inf_ms:.1f}, "v2.1.0-deep-cnn", {list(probs)});
                    }} else {{
                        setTimeout(checkAndRender, 50);
                    }}
                }};
                checkAndRender();
            }})();
            </script>
            """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="features"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-center">
  <div class="section-tag">Capabilities</div>
  <div class="section-title">⚡ Engineered for Excellence</div>
  <div class="section-desc">
    A full-featured AI recognition platform combining deep learning with a premium developer experience.
  </div>
</div>
<div class="feat-grid">
  <div class="feat-card">
    <span class="feat-icon">⚡</span>
    <div class="feat-title">Real-time Inference</div>
    <div class="feat-desc">Sub-millisecond predictions on normalised 28×28 images using optimised TensorFlow/Keras model.</div>
  </div>
  <div class="feat-card">
    <span class="feat-icon">🧠</span>
    <div class="feat-title">CNN Deep Learning</div>
    <div class="feat-desc">Two Conv2D blocks with Batch Normalisation, MaxPooling, and Dropout for robust generalisation.</div>
  </div>
  <div class="feat-card">
    <span class="feat-icon">🎯</span>
    <div class="feat-title">99.40% Accuracy</div>
    <div class="feat-desc">Trained and validated against the standard MNIST 10,000-image test split with full metric reports.</div>
  </div>
  <div class="feat-card">
    <span class="feat-icon">🎨</span>
    <div class="feat-title">Interactive Canvas</div>
    <div class="feat-desc">Native browser drawing pad feeds live stroke data directly into the Python inference pipeline.</div>
  </div>
  <div class="feat-card">
    <span class="feat-icon">📊</span>
    <div class="feat-title">Probability Explorer</div>
    <div class="feat-desc">Animated gradient bars expose full softmax confidence across all 10 digit classes instantly.</div>
  </div>
  <div class="feat-card">
    <span class="feat-icon">📁</span>
    <div class="feat-title">Multi-source Input</div>
    <div class="feat-desc">Three input modes: Live drawing pad, file upload, and quick MNIST reference samples.</div>
  </div>
  <div class="feat-card">
    <span class="feat-icon">🔍</span>
    <div class="feat-title">Preprocessing Pipeline</div>
    <div class="feat-desc">Automatic bounding-box crop, square padding, Lanczos resize, and pixel normalisation built-in.</div>
  </div>
  <div class="feat-card">
    <span class="feat-icon">💼</span>
    <div class="feat-title">Production Architecture</div>
    <div class="feat-desc">Modular src/ package separating config, data loading, model building, preprocessing, and utilities.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-center">
  <div class="section-tag">Pipeline</div>
  <div class="section-title">⚙️ How It Works</div>
  <div class="section-desc">From raw pixels to an intelligent prediction in five clean stages.</div>
</div>
<div class="timeline">
  <div class="timeline-step">
    <div class="tl-num">Step 01</div>
    <span class="tl-icon">✏️</span>
    <div class="tl-title">Draw or Upload</div>
    <div class="tl-desc">User provides a handwritten digit via canvas, upload, or MNIST sample.</div>
  </div>
  <div class="timeline-arrow">➔</div>
  <div class="timeline-step">
    <div class="tl-num">Step 02</div>
    <span class="tl-icon">🔄</span>
    <div class="tl-title">Preprocessing</div>
    <div class="tl-desc">Crop bounding box, pad to square, resize to 28×28, normalise to [0,1].</div>
  </div>
  <div class="timeline-arrow">➔</div>
  <div class="timeline-step">
    <div class="tl-num">Step 03</div>
    <span class="tl-icon">🧠</span>
    <div class="tl-title">CNN Features</div>
    <div class="tl-desc">Two Conv2D + BatchNorm + MaxPool blocks extract spatial features.</div>
  </div>
  <div class="timeline-arrow">➔</div>
  <div class="timeline-step">
    <div class="tl-num">Step 04</div>
    <span class="tl-icon">⚖️</span>
    <div class="tl-title">Dense Classification</div>
    <div class="tl-desc">Flattened features pass through 128-neuron Dense layer with Dropout.</div>
  </div>
  <div class="timeline-arrow">➔</div>
  <div class="timeline-step">
    <div class="tl-num">Step 05</div>
    <span class="tl-icon">🎯</span>
    <div class="tl-title">Softmax Prediction</div>
    <div class="tl-desc">10-class softmax outputs probabilities; argmax gives the final digit.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CNN ARCHITECTURE + HYPERPARAMS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="technology"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-center">
  <div class="section-tag">Architecture</div>
  <div class="section-title">🏗️ Deep CNN Architecture</div>
  <div class="section-desc">A carefully tuned 10-layer convolutional pipeline achieving 99.40% accuracy.</div>
</div>
""", unsafe_allow_html=True)

arch_l, arch_r = st.columns([1.1, 1], gap="large")

with arch_l:
    st.markdown("""
    <div class="gc">
      <div class="gc-title"><div class="gc-title-dot"></div>Layer-by-layer Pipeline</div>
      <div class="cnn-layers">
        <div class="cnn-row">
          <span class="cnn-badge badge-in">Input</span>
          <span class="cnn-info"><strong>28 × 28 × 1</strong> Grayscale image</span>
          <span class="cnn-dim mono">28×28×1</span>
        </div>
        <div class="cnn-arrow">↓</div>
        <div class="cnn-row">
          <span class="cnn-badge badge-cv">Conv2D</span>
          <span class="cnn-info"><strong>32 filters</strong>, 3×3 kernel, ReLU</span>
          <span class="cnn-dim mono">26×26×32</span>
        </div>
        <div class="cnn-arrow">↓</div>
        <div class="cnn-row">
          <span class="cnn-badge badge-bn">BatchNorm</span>
          <span class="cnn-info">Normalise activations, stabilise training</span>
          <span class="cnn-dim mono">26×26×32</span>
        </div>
        <div class="cnn-arrow">↓</div>
        <div class="cnn-row">
          <span class="cnn-badge badge-mp">MaxPool</span>
          <span class="cnn-info">Pool size <strong>2×2</strong> — spatial downsampling</span>
          <span class="cnn-dim mono">13×13×32</span>
        </div>
        <div class="cnn-arrow">↓</div>
        <div class="cnn-row">
          <span class="cnn-badge badge-cv">Conv2D</span>
          <span class="cnn-info"><strong>64 filters</strong>, 3×3 kernel, ReLU</span>
          <span class="cnn-dim mono">11×11×64</span>
        </div>
        <div class="cnn-arrow">↓</div>
        <div class="cnn-row">
          <span class="cnn-badge badge-bn">BatchNorm</span>
          <span class="cnn-info">Improve convergence speed</span>
          <span class="cnn-dim mono">11×11×64</span>
        </div>
        <div class="cnn-arrow">↓</div>
        <div class="cnn-row">
          <span class="cnn-badge badge-mp">MaxPool</span>
          <span class="cnn-info">Pool size <strong>2×2</strong></span>
          <span class="cnn-dim mono">5×5×64</span>
        </div>
        <div class="cnn-arrow">↓</div>
        <div class="cnn-row">
          <span class="cnn-badge badge-fl">Flatten</span>
          <span class="cnn-info">2D → 1D feature vector</span>
          <span class="cnn-dim mono">1600</span>
        </div>
        <div class="cnn-arrow">↓</div>
        <div class="cnn-row">
          <span class="cnn-badge badge-fc">Dense</span>
          <span class="cnn-info"><strong>128 neurons</strong>, ReLU</span>
          <span class="cnn-dim mono">128</span>
        </div>
        <div class="cnn-arrow">↓</div>
        <div class="cnn-row">
          <span class="cnn-badge badge-do">Dropout</span>
          <span class="cnn-info">Rate = <strong>0.3</strong> — prevents overfitting</span>
          <span class="cnn-dim mono">128</span>
        </div>
        <div class="cnn-arrow">↓</div>
        <div class="cnn-row">
          <span class="cnn-badge badge-sm">Softmax</span>
          <span class="cnn-info"><strong>10 neurons</strong> — class probabilities</span>
          <span class="cnn-dim mono">10</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

with arch_r:
    st.markdown("""
    <div class="gc">
      <div class="gc-title"><div class="gc-title-dot"></div>Model Hyperparameters</div>
      <table class="params-table">
        <tr><td>Dataset</td><td>MNIST (60k train / 10k test)</td></tr>
        <tr><td>Framework</td><td style="color:var(--cyan)!important;">TensorFlow / Keras</td></tr>
        <tr><td>Architecture</td><td>Deep 2-Block CNN</td></tr>
        <tr><td>Total Parameters</td><td style="color:var(--highlight)!important;">~110,634</td></tr>
        <tr><td>Optimizer</td><td>Adam (lr = 0.001)</td></tr>
        <tr><td>Loss Function</td><td>Categorical Crossentropy</td></tr>
        <tr><td>Conv Activations</td><td>ReLU</td></tr>
        <tr><td>Output Activation</td><td style="color:var(--success)!important;">Softmax</td></tr>
        <tr><td>Dropout Rate</td><td>0.30</td></tr>
      </table>
      <div style="background:rgba(124,58,237,0.08);border:1px dashed rgba(124,58,237,0.25);border-radius:14px;padding:1rem;margin-top:1.5rem;font-size:0.85rem;color:#c4b5fd;line-height:1.5;">
        ℹ️ MaxPooling reduces the spatial grid from 28×28 → 5×5 before the Dense layer,
        cutting parameters while preserving discriminative features.
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="performance"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-center">
  <div class="section-tag">Analytics</div>
  <div class="section-title">📊 Performance Dashboard</div>
  <div class="section-desc">Verified evaluation metrics on the 10,000-sample MNIST test set.</div>
</div>
<div class="metrics-grid">
  <div class="metric-card"><span class="metric-icon">🎯</span><div class="metric-val">99.40%</div><div class="metric-lbl">Test Accuracy</div></div>
  <div class="metric-card"><span class="metric-icon">⚡</span><div class="metric-val">99.40%</div><div class="metric-lbl">Weighted Precision</div></div>
  <div class="metric-card"><span class="metric-icon">🔁</span><div class="metric-val">99.40%</div><div class="metric-lbl">Weighted Recall</div></div>
  <div class="metric-card"><span class="metric-icon">📐</span><div class="metric-val">99.40%</div><div class="metric-lbl">F1-Score</div></div>
  <div class="metric-card"><span class="metric-icon">⏱️</span><div class="metric-val">~0.8ms</div><div class="metric-lbl">Inference Speed</div></div>
  <div class="metric-card"><span class="metric-icon">🗂️</span><div class="metric-val">70,000</div><div class="metric-lbl">Dataset Images</div></div>
  <div class="metric-card"><span class="metric-icon">🔥</span><div class="metric-val">1.0000</div><div class="metric-lbl">ROC-AUC (OvR)</div></div>
  <div class="metric-card"><span class="metric-icon">🧩</span><div class="metric-val">110K</div><div class="metric-lbl">CNN Parameters</div></div>
</div>
""", unsafe_allow_html=True)

# Interactive chart toggles
st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="gc">', unsafe_allow_html=True)
st.markdown('<div class="gc-title"><div class="gc-title-dot"></div>Evaluation Charts & Diagnostics</div>', unsafe_allow_html=True)
chk1, chk2 = st.columns(2)
with chk1: show_cm = st.checkbox("🔍 Confusion Matrix")
with chk2: show_hist = st.checkbox("📈 Training Curves")
if show_cm:
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    if cm_path.exists(): st.image(str(cm_path), use_container_width=True, caption="Confusion Matrix — MNIST 10-Class Classification")
    else: st.warning("Confusion matrix not found. Run `python evaluate.py` to generate it.")
if show_hist:
    hist_path = REPORTS_DIR / "training_history.png"
    if hist_path.exists(): st.image(str(hist_path), use_container_width=True, caption="Training Loss & Accuracy Convergence Curves")
    else: st.info("Training history graph available after running `python train.py`.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TECH STACK
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-center">
  <div class="section-tag">Stack</div>
  <div class="section-title">💻 Technology Stack</div>
  <div class="section-desc">Industry-standard tools powering DigitVision AI from data pipeline to UI.</div>
</div>
<div class="tech-cloud">
  <div class="tech-pill"><span class="tech-dot" style="background:#3776ab;"></span>Python</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#ff9900;"></span>TensorFlow</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#d00000;"></span>Keras</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#013243;"></span>NumPy</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#5c9c00;"></span>OpenCV</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#009688;"></span>FastAPI</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#e83e8c;"></span>Flask</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#61dafb;"></span>React</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#06b6d4;"></span>Tailwind CSS</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#f7df1e;"></span>JavaScript</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#f05032;"></span>Git</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#6e5494;"></span>GitHub</div>
  <div class="tech-pill"><span class="tech-dot" style="background:#ff6f00;"></span>Streamlit</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT HIGHLIGHTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-center">
  <div class="section-tag">Highlights</div>
  <div class="section-title">🌟 Why DigitVision AI Stands Out</div>
  <div class="section-desc">Key differentiators that make this project production-grade and portfolio-worthy.</div>
</div>
<div class="hl-grid">
  <div class="hl-card hl-green">
    <span class="hl-icon">🎯</span>
    <div class="hl-title">State-of-the-art Accuracy</div>
    <div class="hl-desc">99.40% classification rate on MNIST — matching top academic benchmarks using a clean, minimal architecture.</div>
  </div>
  <div class="hl-card hl-blue">
    <span class="hl-icon">🧠</span>
    <div class="hl-title">Optimised CNN Design</div>
    <div class="hl-desc">Dual Conv2D blocks with Batch Normalisation and Dropout suppress overfitting without sacrificing accuracy.</div>
  </div>
  <div class="hl-card hl-purple">
    <span class="hl-icon">⚡</span>
    <div class="hl-title">Sub-millisecond Inference</div>
    <div class="hl-desc">Optimised weight storage in Keras native format enables real-time prediction for any deployment target.</div>
  </div>
  <div class="hl-card hl-white">
    <span class="hl-icon">🔬</span>
    <div class="hl-title">Research-Inspired</div>
    <div class="hl-desc">Architecture draws from landmark digit recognition papers and modern CNN design best-practices.</div>
  </div>
  <div class="hl-card hl-orange">
    <span class="hl-icon">📁</span>
    <div class="hl-title">Clean Module Architecture</div>
    <div class="hl-desc">Strict separation of concerns: config, data_loader, model_builder, preprocessing, utils, and web UI.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ABOUT THE PROJECT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-center">
  <div class="section-tag">Background</div>
  <div class="section-title">📖 About The Project</div>
  <div class="section-desc">The problem, the solution, the dataset, and what's next.</div>
</div>
<div class="about-grid">
  <div class="about-item">
    <span class="about-item-icon">🔍</span>
    <div class="about-item-title">Problem Statement</div>
    <div class="about-item-text">Handwritten text recognition demands complex feature extraction sensitive to spatial translations, scale, and stroke variations — challenges that rule-based methods fail to handle robustly.</div>
  </div>
  <div class="about-item">
    <span class="about-item-icon">💡</span>
    <div class="about-item-title">Our Solution</div>
    <div class="about-item-text">A multi-layer CNN in TensorFlow/Keras leverages spatial weight sharing and learnt filter matrices to isolate strokes, loops, and edges independently of exact pixel position.</div>
  </div>
  <div class="about-item">
    <span class="about-item-icon">🗃️</span>
    <div class="about-item-title">Dataset</div>
    <div class="about-item-text">MNIST — 70,000 grayscale 28×28 images of handwritten digits 0–9. Split: 54,000 train / 6,000 validation / 10,000 test, loaded directly via TensorFlow Datasets.</div>
  </div>
  <div class="about-item">
    <span class="about-item-icon">📈</span>
    <div class="about-item-title">Results</div>
    <div class="about-item-text">99.40% test accuracy, 1.0000 ROC-AUC (OvR), 99.40% weighted precision, recall, and F1-score across all 10 classes on 10,000 independent test images.</div>
  </div>
  <div class="about-item">
    <span class="about-item-icon">🚀</span>
    <div class="about-item-title">Future Improvements</div>
    <div class="about-item-text">Expand to EMNIST for A–Z recognition, compile the backend to WebAssembly for client-side inference, add data augmentation, and benchmark against EfficientNet.</div>
  </div>
  <div class="about-item">
    <span class="about-item-icon">⚙️</span>
    <div class="about-item-title">Training Pipeline</div>
    <div class="about-item-text">Adam optimiser (lr=0.001), Categorical Crossentropy loss, EarlyStopping on val_loss, ModelCheckpoint saving best weights, ReduceLROnPlateau scheduling.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ABOUT THE DEVELOPER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-center">
  <div class="section-tag">Team</div>
  <div class="section-title">🧑‍💻 Meet the Developer</div>
</div>
<div class="gc" style="margin-top:1.5rem;">
  <div class="dev-wrapper">
    <div class="dev-left">
      <div class="dev-avatar">👨‍💻</div>
      <div class="dev-name">Muhammad Abdullah</div>
      <div class="dev-title">AI & Machine Learning Engineer</div>
      <div class="dev-badges">
        <span class="dev-badge badge-avail">🟢 Available for AI & ML Opportunities</span>
        <span class="dev-badge badge-oss">⭐ Open Source Contributor</span>
        <span class="dev-badge badge-ai">🤖 AI Enthusiast</span>
      </div>
    </div>
    <div class="dev-right">
      <div class="dev-bio">
        Passionate Computer Science student specialising in Artificial Intelligence, Machine Learning,
        Deep Learning, Natural Language Processing, and modern full-stack web development.
        Focused on building scalable AI-powered applications with clean architecture, high performance,
        and exceptional user experience.
      </div>
      <div class="skills-flex">
        <span class="skill-tag">Python</span>
        <span class="skill-tag">Machine Learning</span>
        <span class="skill-tag">Deep Learning</span>
        <span class="skill-tag">CNN</span>
        <span class="skill-tag">TensorFlow</span>
        <span class="skill-tag">Keras</span>
        <span class="skill-tag">OpenCV</span>
        <span class="skill-tag">NLP</span>
        <span class="skill-tag">React</span>
        <span class="skill-tag">FastAPI</span>
        <span class="skill-tag">Tailwind CSS</span>
        <span class="skill-tag">Git</span>
        <span class="skill-tag">GitHub</span>
      </div>
      <div class="dev-socials">
        <a href="https://github.com/muhammadabdullah-devpk" target="_blank" class="social-btn">🐙 GitHub</a>
        <a href="https://linkedin.com/in/muhammad-abdullah-devpk" target="_blank" class="social-btn">💼 LinkedIn</a>
        <a href="mailto:meharabdullah4337@gmail.com" class="social-btn">✉️ Email</a>
      </div>
      <div class="dev-action-btns">
        <a href="https://github.com/muhammadabdullah-devpk" target="_blank" class="dev-btn-primary">View GitHub</a>
        <a href="https://linkedin.com/in/muhammad-abdullah-devpk" target="_blank" class="dev-btn-secondary">Connect on LinkedIn</a>
        <a href="mailto:meharabdullah4337@gmail.com" class="dev-btn-secondary">Contact Me</a>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height:3rem;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-grid">
    <div>
      <div class="footer-logo">👁️ DigitVision AI</div>
      <div class="footer-tagline">
        An advanced handwritten digit recognition platform powered by Deep CNN,
        delivering 99.40% accuracy on the MNIST dataset.
        Built as a portfolio-grade, production-ready AI project.
      </div>
    </div>
    <div>
      <div class="footer-col-title">Quick Links</div>
      <a href="#home"        class="footer-link">Home</a>
      <a href="#demo"        class="footer-link">Live Demo</a>
      <a href="#features"    class="footer-link">Features</a>
      <a href="#performance" class="footer-link">Performance</a>
      <a href="#technology"  class="footer-link">Technology</a>
      <a href="#about"       class="footer-link">About</a>
    </div>
    <div>
      <div class="footer-col-title">Resources</div>
      <a href="https://github.com/muhammadabdullah-devpk" target="_blank" class="footer-link">🐙 GitHub Repository</a>
      <a href="https://linkedin.com/in/muhammad-abdullah-devpk" target="_blank" class="footer-link">💼 LinkedIn Profile</a>
      <a href="mailto:meharabdullah4337@gmail.com" class="footer-link">✉️ Email Me</a>
    </div>
    <div>
      <div class="footer-col-title">Contact</div>
      <div class="footer-contact-item">
        <span class="footer-contact-lbl">Developer</span>
        <span class="footer-contact-val">Muhammad Abdullah</span>
      </div>
      <div class="footer-contact-item">
        <span class="footer-contact-lbl">Title</span>
        <span class="footer-contact-val">AI & Machine Learning Engineer</span>
      </div>
      <div class="footer-contact-item">
        <span class="footer-contact-lbl">GitHub</span>
        <a href="https://github.com/muhammadabdullah-devpk" target="_blank" class="footer-contact-val">github.com/muhammadabdullah-devpk</a>
      </div>
      <div class="footer-contact-item">
        <span class="footer-contact-lbl">LinkedIn</span>
        <a href="https://linkedin.com/in/muhammad-abdullah-devpk" target="_blank" class="footer-contact-val">linkedin.com/in/muhammad-abdullah-devpk</a>
      </div>
      <div class="footer-contact-item">
        <span class="footer-contact-lbl">Email</span>
        <a href="mailto:meharabdullah4337@gmail.com" class="footer-contact-val">meharabdullah4337@gmail.com</a>
      </div>
    </div>
  </div>
  <div class="footer-bottom">
    <div class="footer-copy">© 2026 Muhammad Abdullah. All Rights Reserved.</div>
    <div class="footer-built">
      Designed & Developed by <strong style="color:var(--text2);">Muhammad Abdullah</strong> ·
      Built with ❤️ using TensorFlow, Python, React, Tailwind CSS &amp; FastAPI.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
