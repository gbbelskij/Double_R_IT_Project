import { z } from "zod";

import { validatePasswordStrength } from "@utils/validatePasswordStrength";

const passwordRegex = /^[A-Za-z0-9._\-@#!$%^&*()+=]+$/;

export const profileSchema = z
  .object({
    first_name: z
      .string()
      .optional()
      .refine((val) => val === undefined || val.length >= 2, {
        message: "Имя должно быть не короче 2 символов",
      }),
    last_name: z
      .string()
      .optional()
      .refine((val) => val === undefined || val.length >= 2, {
        message: "Фамилия должна быть не короче 2 символов",
      }),
    date_of_birth: z
      .string()
      .optional()
      .refine((val) => val === undefined || val.trim().length > 0, {
        message: "Дата рождения обязательна",
      }),
    email: z
      .string()
      .optional()
      .refine(
        (val) => val === undefined || z.string().email().safeParse(val).success,
        {
          message: "Введите корректную почту",
        }
      ),
    job_position: z
      .string()
      .optional()
      .refine((val) => val === undefined || val.trim().length > 0, {
        message: "Выберите должность",
      }),
    work_experience: z
      .number({
        required_error: "Укажите опыт",
        invalid_type_error: "Опыт должен быть числом",
      })
      .min(0, "Опыт должен быть от 0 до 99")
      .max(99, "Опыт должен быть от 0 до 99")
      .optional(),
    old_password: z.string().optional(),
    password: z.string().optional(),
    repeatPassword: z.string().optional(),
  })
  .superRefine((data, ctx) => {
    const { password, repeatPassword, old_password } = data;

    if (password) {
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

      if (!old_password || old_password.trim() === "") {
        ctx.addIssue({
          path: ["old_password"],
          message: "Старый пароль обязателен при смене пароля",
          code: z.ZodIssueCode.custom,
        });
      }

      if (repeatPassword !== password) {
        ctx.addIssue({
          path: ["repeatPassword"],
          message: "Пароли не совпадают",
          code: z.ZodIssueCode.custom,
        });
      }
    } else {
      if (old_password && old_password.trim() !== "") {
        ctx.addIssue({
          path: ["old_password"],
          message: "Старый пароль не нужен, если вы не меняете пароль",
          code: z.ZodIssueCode.custom,
        });
      }
    }
  });

export type ProfileFormData = z.infer<typeof profileSchema>;
