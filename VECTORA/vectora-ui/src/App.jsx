import React, { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";

// ════════════════════════════════════════════════════════════
// CONSTANTS & HELPERS
// ════════════════════════════════════════════════════════════
const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8080";
const DIMS = 16;
const COL = {
  cs: "#00d9ff",
  math: "#b388ff",
  food: "#ffb74d",
  sports: "#69f0ae",
  doc: "#a6e3a1",
  default: "#90a4ae",
};
const DIM_COL = Array(4)
  .fill("#00d9ff")
  .concat(Array(4).fill("#b388ff"))
  .concat(Array(4).fill("#ffb74d"))
  .concat(Array(4).fill("#69f0ae"));
