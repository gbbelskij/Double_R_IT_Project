import { Answer } from "types/question";

export const extractAnswer = (answer: Answer): string => {
  return typeof answer === "object" ? Object.keys(answer)[0] : answer;
};
