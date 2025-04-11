import { AnswerEntries } from "@components/MultiStepSurvey/MultiStepSurvey.types";
import { Question } from "types/question";

export interface ProgressNavProps {
  /**
   * Object of questions for the survey.
   */
  questions: Record<string, Question>;
  /**
   * List of IDs of available and selected answers
   */
  answerIDs: string[];
  /**
   * List of available and selected answers
   */
  selectedAnswers: AnswerEntries;
  /**
   * Callback to change the current step in the MultiStepSurvey
   * @param stepIndex - Index of the step to navigate to
   */
  onStepChange: (stepIndex: number) => void;
}
