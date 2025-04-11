import { Answer } from "src/types/question";

export interface RadioButtonProps {
  /**
   * Value associated with the radio button
   */
  value: Answer;
  /**
   * Whether the radio is currently selected
   */
  isSelected: boolean;
  /**
   * Click handler triggered when the radio is toggled
   */
  onClick: (value: Answer) => void;
}
