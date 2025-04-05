export interface CheckboxButtonProps {
  /**
   * Value associated with the checkbox button
   */
  value: string;
  /**
   * Whether the checkbox is currently selected
   */
  isSelected: boolean;
  /**
   * Click handler triggered when the checkbox is toggled
   */
  onClick: (value: string) => void;
}
