export const validatePasswordStrength = (password: string): number => {
  if (password.length === 0) {
    return 0;
  }

  const hasLowercase = /[a-z]/.test(password);
  const hasUppercase = /[A-Z]/.test(password);
  const hasDigit = /\d/.test(password);
  const hasSpecialChar = /[^\w\s]/.test(password);
  const hasValidLength = password.length >= 8;

  if (!hasValidLength) {
    return 1;
  }

  let score = 0;

  if (password.length >= 12) {
    score += 2;
  } else if (password.length >= 8) {
    score += 1;
  }

  const diversityFactors = [
    hasLowercase,
    hasUppercase,
    hasDigit,
    hasSpecialChar,
    password.length > 14,
  ].filter(Boolean).length;

  score += diversityFactors;

  if (score <= 3) {
    return 1;
  } else if (score <= 5) {
    return 2;
  }

  return 3;
};
