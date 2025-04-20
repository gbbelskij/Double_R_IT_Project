import { IconType } from "react-icons";
import { SelectOption } from "./Select.types";

export interface SelectProps {
  /**
   * Options of select
   */
  options: SelectOption[];
  /**
   * Name attribute of the select element
   */
  name: string;
  /**
   * Optional label displayed above or beside the select
   */
  label?: string;
  /**
   * Optional icon to display inside the select field
   */
  icon?: IconType;
  /**
   * Default value of the select, must be in options
   */
  defaultValue?: string;
  /**
   * Placeholder text shown when select is empty
   */
  placeholder?: string;
}
