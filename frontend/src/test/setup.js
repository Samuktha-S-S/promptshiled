import '@testing-library/jest-dom/vitest';
import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// jsdom does not implement these browser APIs
if (!window.matchMedia) {
  window.matchMedia = (query) => ({
    matches: false, media: query, onchange: null,
    addEventListener: () => {}, removeEventListener: () => {},
    addListener: () => {}, removeListener: () => {}, dispatchEvent: () => false,
  });
}
Element.prototype.scrollIntoView = Element.prototype.scrollIntoView || function scrollIntoView() {};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
