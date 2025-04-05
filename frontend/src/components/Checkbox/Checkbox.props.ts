import { IconType } from "react-icons";

export interface CheckboxProps {
  /**
   * Name of the checkbox input
   */
  name: string;
  /**
   * Optional label to display next to the checkbox
   */
  label?: string;
  /**
   * Optional icon to display with the checkbox
   */
  icon?: IconType;
}
