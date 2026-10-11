import { Fragment, type ReactNode } from "react";
import { getLang, type Key } from "./index";
import { en } from "./en";
import { ru } from "./ru";
import { uz } from "./uz";

const DICTS = { uz, en, ru } as const;

/** Like t(), but the {placeholders} may be JSX (bold values, links): returns a node list in the word order of the language. */
export function tn(key: Key, params: Record<string, ReactNode>): ReactNode {
  const template: string = DICTS[getLang()][key] ?? uz[key];
  const parts = template.split(/(\{\w+\})/g);
  return parts.map((p, i) => {
    const m = p.match(/^\{(\w+)\}$/);
    return <Fragment key={i}>{m && m[1] in params ? params[m[1]] : p}</Fragment>;
  });
}
