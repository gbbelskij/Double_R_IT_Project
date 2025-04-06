import { LinkProps } from "react-router";

export interface LogoProps extends Omit<LinkProps, "to"> {
  /**
   * The color of the logo. Optional. Defaults to var(--solitude-100).
   */
  color?: string;
  /**
   * Whether to display text alongside the logo. Optional. Defaults to `false`.
   */
  hasText?: boolean;
  /**
   * The URL the logo should link to. Optional. Defaults to `/`.
   */
  href?: string;
}
