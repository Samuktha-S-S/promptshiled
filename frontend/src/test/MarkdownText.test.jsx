import React from 'react';
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import MarkdownText from '../components/MarkdownText.jsx';

describe('MarkdownText', () => {
  it('renders fenced code blocks, inline code, bold and lists', () => {
    const md = 'Use **this**:\n\n```python\nprint("hi")\n```\n\n- one\n- `two`\n\n1. first\n2. second';
    const { container } = render(<MarkdownText content={md} />);
    expect(container.querySelector('pre.code-block code').textContent).toBe('print("hi")');
    expect(container.querySelector('.code-lang').textContent).toBe('python');
    expect(container.querySelector('strong').textContent).toBe('this');
    expect(container.querySelectorAll('ul li')).toHaveLength(2);
    expect(container.querySelector('.inline-code').textContent).toBe('two');
    expect(container.querySelectorAll('ol li')).toHaveLength(2);
  });

  it('never interprets model output as HTML', () => {
    const { container } = render(<MarkdownText content={'<img src=x onerror="alert(1)"> <script>bad()</script>'} />);
    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelector('script')).toBeNull();
    expect(container.textContent).toContain('<img src=x');
  });
});
