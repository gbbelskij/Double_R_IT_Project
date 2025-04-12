export const declineMonth = (duration: number): string => {
  const lastDigit = duration % 10;
  const lastTwoDigits = duration % 100;

  if (lastTwoDigits >= 11 && lastTwoDigits <= 14) {
    return `${duration} месяцев`;
  }

  if (lastDigit === 1) {
    return `${duration} месяц`;
  }

  if (lastDigit >= 2 && lastDigit <= 4) {
    return `${duration} месяца`;
  }

  return `${duration} месяцев`;
};

export const declineYear = (n: number): string => {
  const lastDigit = n % 10;
  const lastTwoDigits = n % 100;

  if (lastTwoDigits > 10 && lastTwoDigits < 20) {
    return "лет";
  }

  if (lastDigit === 1) {
    return "год";
  }

  if (lastDigit >= 2 && lastDigit <= 4) {
    return "года";
  }

  return "лет";
};
