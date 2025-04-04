import { Preference } from "../../data";

export interface CheckboxButtonGroupProps {
  options: Preference[];
  selectedOptions: string[];
  onChange: (selected: string[]) => void;
}
