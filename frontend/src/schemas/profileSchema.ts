import { z } from "zod";

export const profileSchema = z
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
    old_password: z.string(),
    password: z.string().min(8, "Пароль должен быть хотя бы средний"),
    repeatPassword: z.string().min(8, "Повторите пароль"),
  })
  .refine((data) => data.password === data.repeatPassword, {
    message: "Пароли не совпадают",
    path: ["repeatPassword"],
  });

export type ProfileFormData = z.infer<typeof profileSchema>;
