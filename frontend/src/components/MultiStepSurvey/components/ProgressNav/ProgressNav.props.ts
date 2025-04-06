import { AnswerEntry } from "@components/MultiStepSurvey/MultiStepSurvey.types";

export interface ProgressNavProps {
  /**
   * List of available and selected answers
   */
  selectedAnswers: AnswerEntry[];
  /**
   * Callback to change the current step in the MultiStepSurvey
   * @param stepIndex - Index of the step to navigate to
   */
  onStepChange: (stepIndex: number) => void;
}
