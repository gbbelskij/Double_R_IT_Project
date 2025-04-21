import { IconType } from "react-icons";
import { InputTypes } from "./Input.types";

export interface InputProps {
  /**
   * Type of the input (e.g., text, number, password)
   */
  type: InputTypes;
  /**
   * Name attribute of the input element
   */
  name: string;
  /**
   * Optional label displayed above or beside the input
   */
  label?: string;
  /**
   * Optional icon to display inside the input field
   */
  icon?: IconType;
  /**
   * Default value of the input
   */
  defaultValue?: string;
  /**
   * Placeholder text shown when input is empty
   */
  placeholder?: string;
  /**
   * Optional function that returns a unit string based on the numeric input value
   */
  getUnit?: (value: number) => string;
}
