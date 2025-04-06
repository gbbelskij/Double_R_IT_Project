export interface RadioButtonProps {
  /**
   * Value associated with the radio button
   */
  value: string;
  /**
   * Whether the radio is currently selected
   */
  isSelected: boolean;
  /**
   * Click handler triggered when the radio is toggled
   */
  onClick: (value: string) => void;
}
