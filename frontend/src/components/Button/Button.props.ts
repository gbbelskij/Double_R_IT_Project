import { ReactNode } from "react";
import { ButtonTypes } from "./enums/ButtonTypes";

/**
 * Props for the Button component
 */
export interface ButtonProps {
  /** Button type defining its style and purpose */
  type: ButtonTypes;
  /** Whether the button should take full width of its container */
  fullWidth?: boolean;
  /** Icon displayed on the left side of the button */
  leftIcon?: ReactNode;
  /** Icon displayed on the right side of the button */
  rightIcon?: ReactNode;
  /** Content inside the button (usually text) */
  children: ReactNode;
}
