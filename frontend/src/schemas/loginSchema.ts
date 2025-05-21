import { z } from "zod";

export const loginSchema = z.object({
  email: z.string().email("Введите корректную почту"),
  password: z.string().min(1, "Введите пароль"),
  remember: z.boolean(),
});

export type LoginFormData = z.infer<typeof loginSchema>;
