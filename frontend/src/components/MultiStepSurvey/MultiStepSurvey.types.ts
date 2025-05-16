/**
 * Represents the structure of user responses within the multi-step survey.
 *
 * Each key corresponds to a unique question ID, and its value is either:
 * - a number indicating the index of the selected answer option, or
 * - `null` if no answer has been provided for that question.
 */
export type SurveyData = Record<string, number | null>;

/**
 * Props for the introductory `Intro` component of the MultiStepSurvey.
 */
interface IntroProps {
  /**
   * Callback function that transitions the survey to the specified step index.
   *
   * Typically used to initiate the first step of the survey process.
   *
   * @param stepIndex - The numeric index of the step to navigate to.
   */
  onStepChange: (stepIndex: number) => void;
  /**
   * @todo
   */
  onLogoClick?: () => void;
}

/**
 * Type definition for an `Intro` component used in MultiStepSurvey.
 *
 * This component must:
 * - Accept `IntroProps` as props.
 * - Forward a ref to its root HTML element (typically a `<div>`).
 */
export type Intro = React.ForwardRefExoticComponent<
  IntroProps & React.RefAttributes<HTMLDivElement>
>;

/**
 * Props for the concluding `Outro` component of the MultiStepSurvey.
 */
interface OutroProps {
  /**
   * Callback function triggered when the user exits the Outro step.
   *
   * This is typically used to perform cleanup actions or navigate away
   * from the survey upon completion.
   */
  onExit: Function;
  /**
   * @todo
   */
  onLogoClick?: () => void;
}

/**
 * Type definition for an `Outro` component used in MultiStepSurvey.
 *
 * This component must:
 * - Accept `OutroProps` as props.
 * - Forward a ref to its root HTML element (typically a `<div>`).
 */
export type Outro = React.ForwardRefExoticComponent<
  OutroProps & React.RefAttributes<HTMLDivElement>
>;
