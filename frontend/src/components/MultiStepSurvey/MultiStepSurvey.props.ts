import { Question } from "types/question";
import { Intro, Outro, SurveyData } from "./MultiStepSurvey.types";

export interface MultiStepSurveyProps {
  /**
   * Object of questions for the survey.
   */
  questions: Record<string, Question>;
  /**
   * Optional custom Intro component that is shown before the first question.
   * Must call `onStepChange(0)` to proceed to the first question.
   */
  Intro?: Intro;
  /**
   * Optional custom Outro component that is shown after the last question.
   */
  Outro?: Outro;
  /**
   * Additional user-specific metadata that will be merged with the selected answers
   * and sent to the backend when the "Finish" button is clicked.
   */
  userMeta?: any;
  /**
   * Callback that is triggered when the multi-step survey is fully completed.
   *
   * @param userMeta - metadata provided by the user during the first registration step (e.g., name, email, etc.).
   * @param answers - survey answers collected during the multi-step survey process.
   */
  onComplete?: (answers: SurveyData, userMeta?: any) => void;
  /**
   * @todo
   */
  onExit?: Function;
  /**
   * @todo
   */
  loading?: boolean;
  /**
   * @todo
   */
  onLogoClick?: () => void;
}
