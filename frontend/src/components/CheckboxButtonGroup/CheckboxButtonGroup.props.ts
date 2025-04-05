import { Preference } from "types/preference";

export interface CheckboxButtonGroupProps {
  /**
   * List of available checkbox options
   */
  options: Preference[];
  /**
   * List of currently selected option values
   */
  selectedOptions: string[];
  /**
   * Callback when the selection changes
   */
  onChange: (selected: string[]) => void;
}
