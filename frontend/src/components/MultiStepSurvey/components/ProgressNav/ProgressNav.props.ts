import { SurveyData } from "@components/MultiStepSurvey/MultiStepSurvey.types";
import { Question } from "types/question";

export interface ProgressNavProps {
  /**
   * Object of questions for the survey.
   */
  questions: Record<string, Question>;
  /**
   * List of IDs of available questions.
   */
  questionIDs: string[];
  /**
   * List of available and selected answers.
   */
  surveyData: SurveyData;
  /**
   * Current survey step.
   */
  step: number;
  /**
   * Callback to change the current step in the MultiStepSurvey.
   * @param stepIndex - Index of the step to navigate to.
   */
  onStepChange: (stepIndex: number) => void;
}
