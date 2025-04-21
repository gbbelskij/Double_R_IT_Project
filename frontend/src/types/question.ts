/**
 * A variant of the answer to a question as `string` or `object`, where:
 * - `key` is the text of the answer,
 * - `value` is the ID of the subquestion.
 */
export type Answer = string | Record<string, string>;

export interface Question {
  /**
   * Question text.
   */
  text: string;
  /**
   * A list of possible answers to the question.
   */
  answers: Answer[];
}
