import { z } from "zod";

export const registrationSchema = z
  .object({
    name: z.string().min(2, "Имя должно быть не короче 2 символов"),
    surname: z.string().min(2, "Фамилия должна быть не короче 2 символов"),
    birthday: z.string().min(1, "Дата рождения обязательна"),
    email: z.string().email("Введите корректную почту"),
    post: z.string().min(1, "Выберите должность"),
    experience: z
      .number({
        required_error: "Укажите опыт",
        invalid_type_error: "Опыт должен быть числом",
      })
      .min(0, "Опыт должен быть от 0 до 99")
      .max(99, "Опыт должен быть от 0 до 99"),
    password: z.string().min(8, "Пароль должен быть хотя бы средний"),
    repeatPassword: z.string().min(8, "Повторите пароль"),
  })
  .refine((data) => data.password === data.repeatPassword, {
    message: "Пароли не совпадают",
    path: ["repeatPassword"],
  });

export type RegistrationFormData = z.infer<typeof registrationSchema>;
