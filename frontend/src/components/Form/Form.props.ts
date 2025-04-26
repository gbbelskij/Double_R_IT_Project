import { AlignSelf } from "@components/LogoContainer/LogoContainer.types";

export interface FormProps {
  /**
   * ID of the form.
   */
  id: string;
  /**
   * The content to be rendered inside the `Form` component.
   */
  children: React.ReactNode;
  /**
   * The main title displayed at the top of the form.
   */
  title?: string;
  /**
   * Additional text displayed below the title, used for description or instructions.
   */
  additionalText?: string;
  /**
   * The text displayed on the action button.
   * Defaults to "Submit" if not provided.
   */
  actionText?: string;
  /**
   * The function that will be called when the action button is clicked.
   * Defaults to an empty function.
   */
  handleAction?: (event?: React.FormEvent<HTMLFormElement>) => void;
  /**
   * The flag indicates that the action button is disabled.
   */
  isButtonDisabled?: boolean;
  /**
   * Helper text displayed below the action button, usually used for additional notes or information.
   */
  helperText?: string;
  /**
   * The text for the helper link displayed below the action button.
   */
  helperLinkText?: string;
  /**
   * The URL for the helper link displayed below the action button.
   */
  helperLink?: string;
  /**
   * A flag indicating that the logo is disabled.
   */
  disableLogo?: boolean;
  /**
   * The offset of the logo in the **number** of pixels.
   * If the value is `false` (default), it means it is centered.
   */
  logoOffset?: number;
  /**
   * Defines how the logo is aligned along the cross axis.
   * Defaults to `center`.
   */
  logoAlign?: AlignSelf;
  /**
   * Additional styles for form element.
   */
  formClassName?: string;
  /**
   * Disables the background of the form.
   */
  plain?: boolean;
}
