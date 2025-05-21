import { z } from "zod";

import { validatePasswordStrength } from "@utils/validatePasswordStrength";

const passwordRegex = /^[A-Za-z0-9._\-@#!$%^&*()+=]+$/;

export const registrationSchema = z
  .object({
    first_name: z.string().min(2, "Имя должно быть не короче 2 символов"),
    last_name: z.string().min(2, "Фамилия должна быть не короче 2 символов"),
    date_of_birth: z.string().min(1, "Дата рождения обязательна"),
    email: z.string().email("Введите корректную почту"),
    job_position: z.string().min(1, "Выберите должность"),
    work_experience: z
      .number({
        required_error: "Укажите опыт",
        invalid_type_error: "Опыт должен быть числом",
      })
      .min(0, "Опыт должен быть от 0 до 99")
      .max(99, "Опыт должен быть от 0 до 99"),
    password: z.string(),
    repeatPassword: z.string(),
  })
  .superRefine((data, ctx) => {
    const { password, repeatPassword } = data;

    const strength = validatePasswordStrength(password);

    if (strength < 2) {
      ctx.addIssue({
        path: ["password"],
        message: "Пароль должен быть хотя бы средней сложности",
        code: z.ZodIssueCode.custom,
      });
    }

    if (!passwordRegex.test(password)) {
      ctx.addIssue({
        path: ["password"],
        message:
          "Пароль может содержать латинские буквы, цифры и спецсимволы . _ - @ # ! $ % ^ & * ( ) + =",
        code: z.ZodIssueCode.custom,
      });
    }

    if (password !== repeatPassword) {
      ctx.addIssue({
        path: ["repeatPassword"],
        message: "Пароли не совпадают",
        code: z.ZodIssueCode.custom,
      });
    }
  });

export type RegistrationFormData = z.infer<typeof registrationSchema>;
