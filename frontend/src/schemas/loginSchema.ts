import { z } from "zod";

export const loginSchema = z.object({
  email: z.string().email("Введите корректную почту"),
  password: z.string().min(6, "Пароль должен быть минимум 6 символов"),
  remember: z.boolean().refine((val) => val === true, {
    message: "Вы должны согласиться с условиями",
  }),
});

export type LoginFormData = z.infer<typeof loginSchema>;
