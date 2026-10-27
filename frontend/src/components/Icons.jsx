import React from 'react';

/** Small inline SVG icon set (stroke icons, inherit currentColor). */
function Svg({ children, size = 18, ...props }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" {...props}>
      {children}
    </svg>
  );
}

export const ShieldIcon = (p) => (
  <Svg {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></Svg>
);
export const ShieldCheckIcon = (p) => (
  <Svg {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /><path d="m9 12 2 2 4-4" /></Svg>
);
export const ShieldAlertIcon = (p) => (
  <Svg {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /><path d="M12 8v4" /><path d="M12 16h.01" /></Svg>
);
export const AlertTriangleIcon = (p) => (
  <Svg {...p}><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /><path d="M12 9v4" /><path d="M12 17h.01" /></Svg>
);
export const AlertCircleIcon = (p) => (
  <Svg {...p}><circle cx="12" cy="12" r="10" /><path d="M12 8v4" /><path d="M12 16h.01" /></Svg>
);
export const SendIcon = (p) => (
  <Svg {...p}><path d="M12 19V5" /><path d="m5 12 7-7 7 7" /></Svg>
);
export const PlusIcon = (p) => (
  <Svg {...p}><path d="M12 5v14" /><path d="M5 12h14" /></Svg>
);
export const SidebarIcon = (p) => (
  <Svg {...p}><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M9 3v18" /></Svg>
);
export const MenuIcon = (p) => (
  <Svg {...p}><path d="M4 6h16" /><path d="M4 12h16" /><path d="M4 18h16" /></Svg>
);
export const CloseIcon = (p) => (
  <Svg {...p}><path d="M18 6 6 18" /><path d="m6 6 12 12" /></Svg>
);
export const ChatIcon = (p) => (
  <Svg {...p}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></Svg>
);
export const RefreshIcon = (p) => (
  <Svg {...p}><path d="M21 12a9 9 0 1 1-2.6-6.4" /><path d="M21 3v6h-6" /></Svg>
);
export const ChevronDownIcon = (p) => (
  <Svg {...p}><path d="m6 9 6 6 6-6" /></Svg>
);
export const UserIcon = (p) => (
  <Svg {...p}><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></Svg>
);
