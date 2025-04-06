export type AnswerEntry = {
  question: string;
  answer: string | null;
};

export interface IntroProps {
  /**
   * Callback to change the current step in the MultiStepSurvey.
   * @param stepIndex - Index of the step to navigate to.
   */
  onStepChange: (stepIndex: number) => void;
}
