export interface PasswordStrengthCheckerProps {
  /**
   * The password string to be checked for strength.
   */
  password: string;
  /**
   * A function that receives a password and returns a strength level from 1 to 3.
   */
  validateStrength?: (password: string) => number;
  /**
   * Text shown to guide the user in creating a strong password.
   */
  guidanceText?: string;
}
