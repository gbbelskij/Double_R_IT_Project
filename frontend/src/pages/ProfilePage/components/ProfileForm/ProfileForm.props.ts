export interface ProfileFormProps {
  /**
   * ID of the form.
   */
  id?: string;
  /**
   * The content to be rendered inside the `ProfileForm` component.
   */
  children: React.ReactNode;
  /**
   * The function that will be called when the action button is clicked.
   * Defaults to an empty function.
   */
  handleAction?: (event?: React.FormEvent<HTMLFormElement>) => void;
  /**
   * The function that will be called when the action button is clicked.
   * Defaults to an empty function.
   */
  handleSurvey?: React.MouseEventHandler<HTMLButtonElement>;
  /**
   * The function that will be called when the survey button is clicked.
   * Defaults to an empty function.
   */
  handleExit?: React.MouseEventHandler<HTMLButtonElement>;
  /**
   * The function that will be called when the exit button is clicked.
   * Defaults to an empty function.
   */
  handleDelete?: React.MouseEventHandler<HTMLButtonElement>;
  /**
   * The flag indicates that the delete button is disabled.
   */
  isButtonDisabled?: boolean;
}
