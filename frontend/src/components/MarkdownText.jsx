import React from 'react';

/**
 * Minimal, safe Markdown renderer for assistant replies.
 * Supports fenced code blocks, inline code, **bold**, headings and simple
 * bullet / numbered lists. Output is built from React elements only (no
 * dangerouslySetInnerHTML), so model output cannot inject HTML.
 */

function renderInline(text, keyPrefix) {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return <code key={key} className="inline-code">{part.slice(1, -1)}</code>;
    }
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

function parseBlocks(content) {
  const lines = content.replace(/\r\n/g, '\n').split('\n');
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*```(\S*)/);
    if (fence) {
      const body = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      i += 1; // closing fence (or end of text)
      blocks.push({ type: 'code', lang: fence[1], text: body.join('\n') });
      continue;
    }
    if (/^\s*([-*•])\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*([-*•])\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*[-*•]\s+/, ''));
      blocks.push({ type: 'ul', items });
      continue;
    }
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*\d+[.)]\s+/, ''));
      blocks.push({ type: 'ol', items });
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      blocks.push({ type: 'h', text: heading[2] });
      i += 1;
      continue;
    }
    if (!line.trim()) {
      i += 1;
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^\s*(```|[-*•]\s|\d+[.)]\s|#{1,4}\s)/.test(lines[i])) {
      para.push(lines[i++]);
    }
    blocks.push({ type: 'p', text: para.join('\n') });
  }
  return blocks;
}

export default function MarkdownText({ content }) {
  return (
    <div className="markdown">
      {parseBlocks(content || '').map((block, b) => {
        switch (block.type) {
          case 'code':
            return (
              <pre key={b} className="code-block" data-lang={block.lang || undefined}>
                {block.lang && <span className="code-lang">{block.lang}</span>}
                <code>{block.text}</code>
              </pre>
            );
          case 'ul':
            return <ul key={b}>{block.items.map((t, k) => <li key={k}>{renderInline(t, `${b}-${k}`)}</li>)}</ul>;
          case 'ol':
            return <ol key={b}>{block.items.map((t, k) => <li key={k}>{renderInline(t, `${b}-${k}`)}</li>)}</ol>;
          case 'h':
            return <p key={b} className="md-heading">{renderInline(block.text, `${b}`)}</p>;
          default:
            return (
              <p key={b}>
                {block.text.split('\n').map((ln, k) => (
                  <React.Fragment key={k}>{k > 0 && <br />}{renderInline(ln, `${b}-${k}`)}</React.Fragment>
                ))}
              </p>
            );
        }
      })}
    </div>
  );
}
