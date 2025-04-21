import React from "react";
import { Question } from "types/question";
import { IntroProps } from "./MultiStepSurvey.types";

export interface MultiStepSurveyProps {
  /**
   * Object of questions for the survey.
   */
  questions: Record<string, Question>;
  /**
   * Optional custom Intro component that is shown before the first question.
   * Must call `onStepChange(0)` to proceed to the first question.
   */
  Intro?: React.FC<IntroProps>;
  /**
   * Optional custom Outro component that is shown after the last question.
   */
  Outro?: React.FC;
  /**
   * Additional user-specific metadata that will be merged with the selected answers
   * and sent to the backend when the "Finish" button is clicked.
   */
  userMeta?: JSON | null;
}
