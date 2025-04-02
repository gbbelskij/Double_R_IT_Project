export interface CheckboxButtonGroupProps {
  options: string[];
  selectedOptions: string[];
  onChange: (selected: string[]) => void;
}