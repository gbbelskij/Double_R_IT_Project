import axios from "axios";
import { NavigateFunction } from "react-router-dom";

export function handleErrorNavigation(
  error: unknown,
  navigate: NavigateFunction,
  heading: string = "Возникла ошибка.",
  message: string = "Не удалось выполнить запрос. Попробуйте позже.",
  timeout: number = 0,
  to?: string
) {
  let errorMessage = message;

  if (axios.isAxiosError(error) && error.response?.data?.message) {
    errorMessage = error.response.data.message;
  }

  navigate("/error", {
    state: {
      errorHeading: heading,
      errorText: errorMessage,
      timeout: timeout,
      navigateTo: to,
    },
  });
}
